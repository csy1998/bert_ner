# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: ner_reader
@time: 2019/11/6 10:25

    todo(yuxian) 用nlpc的indexer，并且和inference统一。
"""

import numpy as np
import torch
from typing import Dict, List
from shannon_preprocessor.dataset_reader import DatasetReader
from transformers import BertTokenizer
from bert_ner.utils.logger import logger


TASK2LABELS = {
        "detect": ["0", "1"],
        "ner": ["S-NS", "B-NS", "M-NS", "E-NS",
                "S-NR", "B-NR", "M-NR", "E-NR",
                "S-NT", "B-NT", "M-NT", "E-NT", "O"],
        "ner_v2": ['E-TITLE', 'S-GPE', 'M-ORG', 'M-GPE', 'B-PER', 'M-TITLE',
                   'E-PER', 'E-GPE', 'B-TITLE', 'M-PER', 'B-ORG', 'B-GPE', 'E-ORG', 'O'],
        # "ner_title_only": ['B-TITLE', 'E-TITLE', 'M-TITLE', 'O'],
        "cws": ["B-W", "M-W", "E-W", "S-W"],
        "sb_ner": ['(', ')', '()',
                   '((', '))', '(()', '())',
                   '(((', ')))', '((()', '()))',
                   '|', '||', '|||', 'O'],
        "ner_merge": ["S-NS", "B-NS", "M-NS", "E-NS",
                      "S-NR", "B-NR", "M-NR", "E-NR",
                      "S-NT", "B-NT", "M-NT", "E-NT",
                      "S-NP", "B-NP", "M-NP", "E-NP", "O"],
    }


@DatasetReader.register("sequence_labeling")
class SequenceLabelingReader(DatasetReader):
    """
    读取Sequence Labeling tsv的reader
    """

    def __init__(self, args):
        super().__init__(args)
        print("args: ", args)
        self.task2labels = TASK2LABELS
        self.label_list = self.task2labels[args.task]
        self.label_list.extend(["X", "[CLS]", "[SEP]", "[PAD]"])
        self.label_map = {label: i for i, label in enumerate(self.label_list)}
        print(self.label_map)

        self.instance_index = 0
        self.max_seq_length = 128
        self.n_examples = args.n_examples

        if args.tokenizer_type == 'zh':
            bert_path = "/data/nfsdata2/nlp_application/models/bert/chinese_L-12_H-768_A-12"
        elif args.tokenizer_type == 'en':
            bert_path = "/data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-12_H-768_A-12"
        self.tokenizer = BertTokenizer.from_pretrained(bert_path,
                                                       do_lower_case=False)

    @staticmethod
    def add_args(parser):
        """Add specific arguments to the dataset reader."""
        parser.add_argument('--task', type=str, default='ner',
                            help='task to get label_list.')
        parser.add_argument('--n_examples', type=int, default=1,
                            help='print n specific examples of sequence labeling')
        parser.add_argument('--tokenizer_type', type=str, default='zh',
                            help='tokenizer type for different dataset')

    def get_inputs(self, line: str) -> Dict[str, torch.Tensor]:
        """get input from file"""
        words, labels = line.split("\t")
        words = words.split()
        labels = labels.split()
        tokens, labels, label_mask = self.sequence_labeling_tokenize(words, labels)

        input_ids, label_ids, label_mask, segment_ids, attention_mask = self.index(tokens, labels, label_mask)
        return {
            "inputs": torch.IntTensor(input_ids),
            "labels": torch.IntTensor(label_ids),
            "label_mask": torch.IntTensor(label_mask),
            "segment_ids": torch.IntTensor(segment_ids),
            "attention_mask": torch.IntTensor(attention_mask),
        }

    @property
    def fields2dtypes(self):
        """define numpy dtypes of each field."""
        return {
            "inputs": np.uint16,  # 注意当int超过65500时候就不能用uint16了
            "labels": np.uint16,
            "label_mask": np.uint16,
            "segment_ids": np.uint16,
            "attention_mask": np.uint16
        }

    def sequence_labeling_tokenize(self, words: List[str], word_labels: List[str]):
        """tokenize and add label X for bpe"""  # todo 没有UNK？
        tokens = []
        token_labels = []
        masks = []
        for word, label in zip(words, word_labels):
            tmp_tokens = self.tokenizer.tokenize(word)
            try:
                tokens.append(tmp_tokens[0])
                token_labels.append(label)
                masks.append(1)
                for token in tmp_tokens[1:]:
                    tokens.append(token)
                    token_labels.append("X")
                    masks.append(0)
            except IndexError:
                print(words)
                tokens.append("[UNK]")
                token_labels.append(label)
                masks.append(1)
        return tokens, token_labels, masks

    def index(self, tokens, labels, label_mask,
                cls_token_at_end=False, pad_on_left=False,
                cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                sequence_a_segment_id=0, sequence_b_segment_id=1,
                cls_token_segment_id=0, pad_token_segment_id=0,
                mask_padding_with_zero=True
                ):
        """str2int todo 支持变长的存储"""
        max_seq_length = self.max_seq_length
        label_map = self.label_map

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]
            labels = labels[:(max_seq_length - 2)]
            label_mask = label_mask[:(max_seq_length - 2)]

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
        tokens = tokens + [sep_token]
        labels = labels + ['[SEP]']
        label_mask = label_mask + [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            labels = labels + ['[CLS]']
            label_mask = label_mask + [0]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            labels = ['[CLS]'] + labels
            label_mask = [0] + label_mask
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        label_ids = [label_map[x] for x in labels]


        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            label_ids = ([label_map["[PAD]"]] * padding_length) + label_ids
            label_mask = ([0] * padding_length) + label_mask
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            label_ids = label_ids + ([label_map["[PAD]"]] * padding_length)
            label_mask = label_mask + ([0] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(label_mask) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if self.instance_index < self.n_examples:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("label: %s" % " ".join([str(x) for x in label_ids]))
            logger.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        self.instance_index += 1
        return input_ids, label_ids, label_mask, segment_ids, attention_mask
