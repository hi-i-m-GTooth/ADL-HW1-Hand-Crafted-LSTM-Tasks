import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from tqdm import trange

import torch

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab
from torch.utils.data import DataLoader


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    """
    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    
    model.eval()
    """

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    device = args.device
    model = ckpt.to(device)
    # TODO: predict dataset
    test_data = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, collate_fn=dataset.collate_fn)
    # TODO: write prediction to file (args.pred_file)
    f = open(args.pred_file, 'w')
    f.write("id,intent\n")
    # TODO: Training loop - iterate over train dataloader and update model weights
    model.eval()
    for ins_batch in test_data:
        ids = ins_batch["id"]
        texts, text_lens = torch.tensor(ins_batch["textids"]).to(device), torch.tensor(ins_batch["text_len"]).to("cpu")
        ys = model(texts, text_lens)
        
        (_, y_pred) = torch.max(ys, 1)
        for i in range(len(ids)):
            f.write(f"{ids[i]},{dataset.idx2label(y_pred[i].item())}\n")
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
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

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
