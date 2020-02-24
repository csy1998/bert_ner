# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: tagger
@time: 2019/11/19 17:49

    这一行开始写关于本文件的说明与解释
"""

import os
from typing import List

import torch
import torch.nn.functional as F
from service_streamer import Streamer

from bert_ner.deploy.streamer import ManagedLabelingModel
from bert_ner.utils.bmes_decode import bmes_decode, Tag
from transformers import BertTokenizer


class Tagger:
    """Deploy Tagger Interface"""

    def __init__(self,
                 task: str,
                 model_path: str,
                 max_length: int = 128,
                 batch_size: int = 32,
                 cuda_devices: List[int] = None,
                 worker_num: int = 1,
                 deploy_method: str = "streamer"):

        self.task = task
        self.threshold = 0.98
        self.model_path = model_path

        if deploy_method == "streamer":
            self.model = Streamer(ManagedLabelingModel, batch_size=batch_size, max_latency=0.1, worker_num=worker_num,
                                  cuda_devices=cuda_devices or [0], model_init_kwargs={"model_path": model_path,
                                                                                       "max_length": max_length})
        else:
            self.model = ManagedLabelingModel("".join(str(gpu_id) for gpu_id in cuda_devices))
            self.model.init_model(model_path=model_path, max_length=max_length)

        self.label_map = self.load_label_map(model_path=model_path)
        self.max_length = max_length
        self.indexer = BertTokenizer.from_pretrained(model_path)
        print(self.label_map)

    def batch_ner(self, raw_sentences: List[str], input_ids: List[List[int]] = None) -> List[List[Tag]]:
        """
        :param input_ids: input tensor
        :param raw_sentences: raw strings
        :return: [[("北", "B-W"), ("京","E-W"), ...], [("我","S-W"), ...] ...]
        """

        if input_ids is None:
            input_ids = self._index_all_sentences(raw_sentences=raw_sentences)

        if len(raw_sentences) != len(input_ids):
            raise ValueError(f"length of raw_sentences {len(raw_sentences)} is "
                             f"inconsistent with input_ids' batch_size {len(input_ids)}")
        # inference
        outputs = self.model.predict(input_ids)
        # B * L * (num_label+4)     4: ["X", "[CLS]", "[SEP]", "[PAD]"]
        logits = torch.stack(outputs).squeeze(1)[:, :, :len(self.label_map) + 1]
        label_ids = torch.argmax(logits, dim=-1)

        if self.task == 'detect':
            probs = F.softmax(logits, dim=-1)
            label_ids = (probs[:, :, 1] >= self.threshold).int()

        # format
        term_labels = []
        for ids, raw_sentence in zip(label_ids, raw_sentences):
            tmp = []
            for i, idx in enumerate(ids):
                if i == 0:  # [CLS]
                    continue
                if i == len(raw_sentence) + 1:  # [SEP]
                    break
                tmp.append((raw_sentence[i - 1], self.label_map[idx.item()]))
            term_labels.append(tmp)

        # decode
        if self.task in ['ner', 'cws']:
            tags_list: List[List[Tag]] = [bmes_decode(term_label)[1] for term_label in term_labels]
            return tags_list
        elif self.task == 'detect':
            results = []
            for ids, raw_sentence in zip(label_ids, raw_sentences):
                ids = ids.numpy()[1:-1][:len(raw_sentence)]
                sent_list = list(raw_sentence)
                sent_list = [sent_list[i] if label == 0 else '【*' + sent_list[i] + '*】' for i, label in enumerate(ids)]
                results.append(' '.join(sent_list))
            return results

    @staticmethod
    def load_label_map(model_path):
        """load label map from model_dir"""
        label_map_path = os.path.join(model_path, "label_map.txt")
        count = 0
        label_map = {}
        with open(label_map_path, 'r') as label_file:
            for line in label_file:
                label = line.strip()
                label_map[count] = label
                count += 1
        return label_map

    def _index_sentence(self, sentence: str) -> List[int]:
        """tokenize todo(yuxian): 用nlpc的"""
        tokens = list(sentence)
        # 长句截断，注意只影响input_ids及模型输出, batch_raw_sentences未截断
        if len(tokens) >= self.max_length - 2:
            tokens = tokens[:self.max_length - 2]
        # add cls and sep
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        # padding
        tokens += ['[PAD]'] * (self.max_length - len(tokens))
        # convert to ids
        ids = self.indexer.convert_tokens_to_ids(tokens)
        return ids

    def _index_all_sentences(self, raw_sentences: List[str]) -> List[List[int]]:
        """convert list of sentence into list of token ids"""
        input_ids = []
        for sentence in raw_sentences:
            indexed_text = self._index_sentence(sentence)
            input_ids.append(indexed_text)
        return input_ids
