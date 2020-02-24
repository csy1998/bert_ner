# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: compressor
@time: 2019/11/21 14:18

    这一行开始写关于本文件的说明与解释
"""


import os
from typing import List, Tuple
from math import ceil, floor

import torch
from service_streamer import Streamer

from bert_ner.deploy.streamer import ManagedLabelingModel
from pytorch_transformers import BertTokenizer


class Compressor:
    """Deploy Tagger Interface"""

    def __init__(self, model_path: str, max_length: int = 128, batch_size: int = 32,
                 cuda_devices: List[int] = None, worker_num: int = 1, deploy_method: str = "streamer"):
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

    def forward(self, raw_sentences: List[str], input_ids: List[List[int]] = None):
        """run model forward to get logits"""
        if input_ids is None:
            input_ids = self._index_all_sentences(raw_sentences=raw_sentences)

        if len(raw_sentences) != len(input_ids):
            raise ValueError(f"length of raw_sentences {len(raw_sentences)} is "
                             f"inconsistent with input_ids' batch_size {len(input_ids)}")
        # inference
        outputs = self.model.predict(input_ids)
        logits = torch.stack(outputs).squeeze(1)[:, :, :len(self.label_map) + 1]
        return logits

    def decode_label_from_logits(self, logits, raw_sentences, ratio_range: Tuple[float, float] = None) -> List[List[int]]:
        """decode label from logits that come from forward function"""
        probs = torch.softmax(logits, dim=2)[:, :, 1]  # 每个字被保留的概率

        if ratio_range is None:
            thresholds = [0.5] * len(raw_sentences)
        else:
            thresholds = [self.find_best_cut(prob[1:len(raw_sentences)+1], ratio_range) for prob, raw_sentences in
                          zip(probs, raw_sentences)]

        # label_ids = torch.argmax(logits, dim=2)
        # format
        term_labels = []
        for sent_prob, raw_sentence, sent_thresh in zip(probs, raw_sentences, thresholds):
            tmp = []
            for i, prob in enumerate(sent_prob):
                if i == 0:  # [CLS]
                    continue
                if i == len(raw_sentence) + 1:  # [SEP]
                    break
                label = 0 if prob < sent_thresh else 1
                tmp.append(label)
            term_labels.append(tmp)

        return term_labels

    def batch_compress(self, raw_sentences: List[str], input_ids: List[List[int]] = None,
                       ratio_range: Tuple[float, float] = None) -> List[List[int]]:
        """
        :param input_ids: input tensor
        :param raw_sentences: raw strings
        :param ratio_range: 指定压缩比例的上下限
        :return: [[("北", "B-W"), ("京","E-W"), ...], [("我","S-W"), ...] ...]
        """

        logits = self.forward(raw_sentences=raw_sentences, input_ids=input_ids)
        term_labels = self.decode_label_from_logits(logits=logits, raw_sentences=raw_sentences,
                                                    ratio_range=ratio_range)

        return term_labels

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

    @staticmethod
    def find_best_cut(probs: List[float], ratio_range: Tuple[float, float]) -> float:
        """根据probs的分布和指定的压缩率范围寻找最稳健的切割点，也即增加后最大的点。返回的是threshold，也即保留大于等于
        threshold的字"""
        idx_probs = [(idx, prob) for idx, prob in enumerate(probs)]
        idx_probs = sorted(idx_probs, key=lambda x: x[1])
        start = ceil(len(probs) * ratio_range[0])
        end = floor(len(probs) * ratio_range[1])
        # best_idx = None
        best_th = 0.5
        best_delta = 0
        for cut_idx in range(max(start, 1), min(end, len(probs)-1)):
            delta = idx_probs[cut_idx][1] - idx_probs[cut_idx - 1][1]
            if delta > best_delta:
                best_delta = delta
                # best_idx = cut_idx
                best_th = idx_probs[cut_idx][1]
        return min(best_th, 0.5)   # 一定要保留大于0.5的
