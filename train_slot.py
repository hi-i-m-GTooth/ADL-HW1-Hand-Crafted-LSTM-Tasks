import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

import torch.nn as nn
import os

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    # TODO: implement main function
    # raise NotImplementedError
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "tag2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    # data = list of {
    # "text": "i need you to book me a flight from ft lauderdale to houston on southwest",
    # "intent": "book_flight",
    # "id": "train-0"
    # },
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_data = DataLoader(datasets[TRAIN], batch_size = args.batch_size, shuffle = True, collate_fn=datasets[TRAIN].collate_fn)
    dev_data = DataLoader(datasets[DEV], batch_size = args.batch_size, shuffle = False, collate_fn=datasets[DEV].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    device = args.device
    model = SeqTagger(embeddings, args.hidden_size, args.num_layers, args.dropout, bool(args.bidirectional), datasets[TRAIN].num_classes).to(device)

    # TODO: init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.sch_gamma)

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        all_pred, correct_pred = 0, 0
        for ins_batch in train_data:
            w = torch.tensor(ins_batch["loss_w"]).to(device)
            labels = torch.tensor(ins_batch["label"]).to(device)
            texts, text_lens = torch.tensor(ins_batch["textids"]).to(device), torch.tensor(ins_batch["text_len"]).to("cpu")
            ys = model(texts, text_lens)
            
            (_, y_pred), (_, lab_pred) = torch.max(ys, 2), torch.max(labels, 2)
            #print(lab_pred)
            all_pred += len(ins_batch)
            correct_pred += sum([torch.equal(y_pred[i][w[i][:]], lab_pred[i][w[i][:]]) for i in range(len(ins_batch))])

            ys = ys.view(-1, datasets[TRAIN].num_classes)[w.view(-1,1).squeeze(1)[:]]
            labels = labels.view(-1, datasets[TRAIN].num_classes)[w.view(-1,1).squeeze(1)[:]]
            train_loss = criterion(ys, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
    
        train_acc = correct_pred / all_pred

        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        all_pred, correct_pred = 0, 0
        with torch.no_grad():
            for ins_batch in dev_data:
                w = torch.tensor(ins_batch["loss_w"]).to(device)
                labels = torch.tensor(ins_batch["label"]).to(device)
                texts, text_lens = torch.tensor(ins_batch["textids"]).to(device), torch.tensor(ins_batch["text_len"]).to(device)
                ys = model(texts, text_lens)

                (_, y_pred), (_, lab_pred) = torch.max(ys, 2), torch.max(labels, 2)
                all_pred += len(ins_batch)
                correct_pred += sum([torch.equal(y_pred[i][w[i][:]], lab_pred[i][w[i][:]]) for i in range(len(ins_batch))])

                ys = ys.view(-1, datasets[TRAIN].num_classes)[w.view(-1,1).squeeze(1)[:]]
                labels = labels.view(-1, datasets[TRAIN].num_classes)[w.view(-1,1).squeeze(1)[:]]
                dev_loss = criterion(ys, labels)

            dev_acc = correct_pred / all_pred
        
        if (epoch+1) % args.save_ep == 0:
            full_path = os.path.join(args.ckpt_dir, f"model_{epoch+1}ep.ckpt")
            torch.save(model, full_path)
        
        if (epoch+1) % args.sch_ep == 0:
            scheduler.step()
        _record = f"Epoch [{epoch+1}/{args.num_epoch}] || Train Loss: {train_loss.item():.4f}, Train Acc: {train_acc:.4f} || Val Loss: {dev_loss.item():.4f}, Val Acc: {dev_acc:.4f} || lr: {scheduler.get_lr()}"
        print(_record)
        record = open(os.path.join(args.ckpt_dir, "record.txt"), 'w')
        record.write(_record+'\n')
        record.close()
        #pass
        full_path = os.path.join(args.ckpt_dir, f"model_final.ckpt")
        torch.save(model, full_path)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=int, default=1)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    parser.add_argument("--save_ep", type=int, default=1)
    parser.add_argument("--sch_ep", type=int, default=5)
    parser.add_argument("--sch_gamma", type=float, default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)