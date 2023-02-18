import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab


def main(args):
    # TODO: implement main function
    #raise NotImplementedError
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "tag2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    device = args.device
    model = ckpt.to(device)
    # TODO: predict dataset
    test_data = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, collate_fn=dataset.collate_fn)
    # TODO: write prediction to file (args.pred_file)
    f = open(args.pred_file, 'w')
    f.write("id,tags\n")
    # TODO: Training loop - iterate over train dataloader and update model weights
    model.eval()
    for ins_batch in test_data:
        ids = ins_batch["id"]
        #w = torch.tensor(ins_batch["loss_w"]).to(device)
        texts, text_lens = torch.tensor(ins_batch["textids"]).to(device), torch.tensor(ins_batch["text_len"]).to(device)
        ys = model(texts, text_lens)
        (_, y_pred) = torch.max(ys, 2)
    
        for i in range(len(ids)):
            f.write(f"{ids[i]},{' '.join([dataset.idx2label(tk.item()) for tk in y_pred[i][:text_lens[i]]])}\n")
        #print(lab_pred)
    f.close()


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
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
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)