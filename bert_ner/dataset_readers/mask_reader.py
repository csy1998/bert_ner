# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: ner_reader
@time: 2019/11/6 10:25

    todo(yuxian) 用nlpc的indexer，并且和inference统一。
"""

import torch
import collections
import numpy as np
from typing import Dict, List
from random import random, shuffle, choice

from shannon_preprocessor.dataset_reader import DatasetReader
from transformers import BertTokenizer
from bert_ner.utils.logger import logger


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


@DatasetReader.register("sequence_masking")
class SequenceMaskingReader(DatasetReader):
    """
    对 pretrained 的数据进行 mask 处理
    """
    def __init__(self, args):
        super().__init__(args)
        print("args: ", args)

        self.instance_index = 0
        self.max_seq_length = args.max_seq_len
        self.masked_lm_prob = args.masked_lm_prob
        self.max_predictions_per_seq = args.max_predictions_per_seq

        if args.tokenizer_type == 'zh':
            bert_path = "/data/nfsdata2/nlp_application/models/bert/chinese_L-12_H-768_A-12"
        elif args.tokenizer_type == 'en':
            bert_path = "/data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-12_H-768_A-12"
        self.tokenizer = BertTokenizer.from_pretrained(bert_path,
                                                       do_lower_case=False)
        self.vocab_list = list(self.tokenizer.vocab.keys())

    @staticmethod
    def add_args(parser):
        """Add specific arguments to the dataset reader."""
        parser.add_argument("--max_seq_len", type=int, default=128)
        parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                            help="Probability of masking each token for the LM examples")
        parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                            help="Maximum number of tokens to mask in each sequence")
        parser.add_argument('--tokenizer_type', type=str, default='zh',
                            help='tokenizer type for different dataset')

    def get_inputs(self, line: str) -> Dict[str, torch.Tensor]:
        """get input from file"""
        line = line.strip()
        tokens = self.sequence_labeling_tokenize([line])
        tokens = tokens[:self.max_seq_length-2]
        # print(tokens)
        # ["[CLS]"] + tokens + ["[SEP]"]
        tokens, masked_lm_positions, masked_lm_labels = self.create_masked_lm_predictions(tokens)
        input_ids, lm_label_ids, attention_mask = self.index(tokens, masked_lm_positions, masked_lm_labels)
        return {
            "inputs": torch.IntTensor(input_ids),
            "lm_label_ids": torch.IntTensor(lm_label_ids),
            "attention_mask": torch.IntTensor(attention_mask),
        }

    @property
    def fields2dtypes(self):
        """
        define numpy dtypes of each field.
        """
        return {
            "inputs": np.uint16,  # 注意当int超过65500时候就不能用uint16了
            "lm_label_ids": np.int16,   # 注意 lm_label_ids 中有-1, 所以用 int16
            "attention_mask": np.uint16,
        }

    def sequence_labeling_tokenize(self, words: List[str]):
        """tokenize and add label X for bpe"""  # todo 没有UNK？
        tokens = []
        for word in words:
            tmp_tokens = self.tokenizer.tokenize(word)
            try:
                tokens.append(tmp_tokens[0])
                for token in tmp_tokens[1:]:
                    tokens.append(token)
            except IndexError:
                print("-----", words, "*****", word)
                tokens.append("[UNK]")
        return tokens

    def create_masked_lm_predictions(self, tokens):
        """
        Creates the predictions for the masked LM objective.
        This is mostly copied from the Google BERT repo, but
        with several refactors to clean it up and remove a lot of unnecessary variables.
        input:
            tokens:                 List[str]
        return:
            tokens:                 ["[CLS]"] + tokens + ["[SEP]"]
            mask_indices:           [index] * num_masked
            masked_token_labels:    [masked_word] * num_masked
        """
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        cand_indices = []
        for (i, token) in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue
            cand_indices.append([i])
        shuffle(cand_indices)

        num_to_mask = min(self.max_predictions_per_seq,
                          max(1, int(round(len(tokens) * self.masked_lm_prob))))

        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indices:
            if len(masked_lms) >= num_to_mask:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_mask:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_token = None
                # 80% of the time, replace with [MASK]
                if random() < 0.8:
                    masked_token = "[MASK]"
                else:
                    # 10% of the time, keep original
                    if random() < 0.5:
                        masked_token = tokens[index]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = choice(self.vocab_list)
                masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
                tokens[index] = masked_token

        assert len(masked_lms) <= num_to_mask
        masked_lms = sorted(masked_lms, key=lambda x: x.index)
        mask_indices = [p.index for p in masked_lms]
        masked_token_labels = [p.label for p in masked_lms]

        return tokens, mask_indices, masked_token_labels

    def index(self, tokens,
              masked_lm_positions,
              masked_lm_labels,
              pad_on_left=False,
              pad_token=0,
              mask_padding_with_zero=True):
        """
        str2int todo 支持变长的存储
        input:
            tokens: with cls and sep
        """
        if len(tokens) > self.max_seq_length:
            print(tokens)
        assert len(tokens) <= self.max_seq_length

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        masked_lm_labels = self.tokenizer.convert_tokens_to_ids(masked_lm_labels)
        lm_label_ids = np.array([-1] * self.max_seq_length)
        lm_label_ids[masked_lm_positions] = masked_lm_labels

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

        assert len(input_ids) == self.max_seq_length
        assert len(lm_label_ids) == self.max_seq_length
        assert len(attention_mask) == self.max_seq_length

        if self.instance_index < 1:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("lm_label_ids: %s" % " ".join([str(x) for x in lm_label_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        self.instance_index += 1

        return input_ids, lm_label_ids, attention_mask
