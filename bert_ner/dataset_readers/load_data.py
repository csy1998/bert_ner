# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: load_data
@time: 2019/11/6 14:49
"""

import os
from torch.utils.data import Dataset
from fairseq.data.indexed_dataset import MMapIndexedDataset


class SequenceLabelingDataset(Dataset):
    """Sequence Labeling Dataset"""
    def __init__(self, directory, prefix, fields=None):
        super().__init__()
        fields = fields or ["inputs", "labels", "label_mask", "attention_mask", "segment_ids"]
        self.fields2datasets = {}
        self.fields = fields
        for field in fields:
            self.fields2datasets[field] = MMapIndexedDataset(os.path.join(directory, f"{prefix}.{field}"))
        self.lengths = []
        self.get_lengths()

    def __len__(self):
        return len(self.fields2datasets[self.fields[0]])

    def __getitem__(self, item):
        return [self.fields2datasets[field][item] for field in self.fields]

    def get_lengths(self):
        """
        for group batch sampler
        """
        for label_mask in self.fields2datasets['label_mask']:
            self.lengths.append(label_mask.sum().item())


class SequenceMaskingDataset(Dataset):
    """Sequence Labeling Dataset"""
    def __init__(self, directory, prefix, fields=None):
        super().__init__()
        fields = fields or ["inputs",  "attention_mask", "lm_label_ids"]
        self.fields2datasets = {}
        self.fields = fields
        for field in fields:
            self.fields2datasets[field] = MMapIndexedDataset(os.path.join(directory, f"{prefix}.{field}"))
        self.lengths = []
        self.get_lengths()

    def __len__(self):
        return len(self.fields2datasets[self.fields[0]])

    def __getitem__(self, item):
        return [self.fields2datasets[field][item] for field in self.fields]

    def get_lengths(self):
        """
        for group batch sampler
        """
        for label_mask in self.fields2datasets['attention_mask']:
            self.lengths.append(label_mask.sum().item())


def run():
    path = "/data/nfsdata2/nlp_application/datasets/grammar-correction/chinese/chinese_ner/v2_20191119/bin"
    prefix = "train"
    fields = ["inputs", "labels", "label_mask", "segment_ids"]
    fields = None
    dataset = SequenceLabelingDataset(path, prefix=prefix, fields=fields)
    print(len(dataset))
    for d in dataset:
        print([v.shape for v in d])


if __name__ == '__main__':
    run()
