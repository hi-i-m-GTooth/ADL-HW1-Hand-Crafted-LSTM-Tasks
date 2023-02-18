from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab, pad_to_len


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        instance['text_len'] = len(instance['text'].split(' '))
        instance['textids'] = [ self.vocab.token_to_id(token) for token in instance['text'].split(' ')]
        instance['label'] = [ 0 for i in range(self.num_classes)]
        if "intent" in instance:
            instance['label'][self.label2idx(instance["intent"])] = 1.0
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        #raise NotImplementedError
        rlt = {"textids": pad_to_len([d['textids'] for d in samples], self.max_len, self.vocab.pad_id),
               "label": [d["label"] for d in samples],
               "text_len": [d["text_len"] for d in samples],
               "id": [d["id"] for d in samples]}
        return rlt

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        instance['text_len'] = len(instance['tokens'])
        instance['textids'] = [self.vocab.token_to_id(token) for token in instance['tokens']]
        instance['label'] = []
        instance['loss_w'] = []

    
        if "tags" in instance:
            instance['label'] = []
            instance['loss_w'] = ([True]*instance['text_len'] + [False]*(self.max_len - instance['text_len'])) if self.max_len >= instance['text_len'] else ([1]*self.max_len)
            for t in instance["tags"][:self.max_len]:
                tmp =  [0 for i in range(self.num_classes)]
                tmp[self.label2idx(t)] = 1.0
                instance['label'].append(tmp)
            while(len(instance['label'])<self.max_len):
                tmp =  [0 for i in range(self.num_classes)]
                instance['label'].append(tmp)
        return instance

    def collate_fn(self, samples):
        # TODO: implement collate_fn
        # raise NotImplementedError
        rlt = {"textids": pad_to_len([d['textids'] for d in samples], self.max_len, self.vocab.pad_id),
               # d[label] = maxlen * num class
               "label": [[token_lab for token_lab in d["label"]] for d in samples],
               "text_len": [d["text_len"] for d in samples],
               "id": [d["id"] for d in samples],
               "loss_w": [d["loss_w"] for d in samples]}
        return rlt
