# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: train_utils
@time: 2019/11/19 17:16

    这一行开始写关于本文件的说明与解释
"""

from __future__ import absolute_import, division, print_function
from bert_ner.metrics.span_f1 import mask_span_f1
from bert_ner.metrics.f_measure import f_measure

import csv
import logging
import os
from io import open
import torch
import numpy as np

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, label_ids, label_mask, attention_mask, segment_ids):
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.label_mask = label_mask
        self.attention_mask = attention_mask
        self.segment_ids = segment_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as file:
            reader = csv.reader(file, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class DetectProcessor(DataProcessor):
    """Processor for the Detect data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples


class CwsProcessor(DataProcessor):
    """Processor for the Detect data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["B-W", "M-W", "E-W", "S-W"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples


class NerProcessor(DataProcessor):
    """Processor for the Detect data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["S-NS", "B-NS", "M-NS", "E-NS", "S-NR", "B-NR", "M-NR", "E-NR", "S-NT", "B-NT", "M-NT", "E-NT", "O"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples


class Nerv2Processor(DataProcessor):
    """Processor for the Detect data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ['E-TITLE', 'S-GPE', 'M-ORG', 'M-GPE', 'B-PER', 'M-TITLE',
                'E-PER', 'E-GPE', 'B-TITLE', 'M-PER', 'B-ORG', 'B-GPE', 'E-ORG', 'O']
        # return ['B-TITLE', 'E-TITLE', 'M-TITLE', 'O']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples


def sequence_labeling_tokenize(words, word_labels, tokenizer):
    """tokenize and add label X for bpe"""  # todo 没有UNK？
    tokens = []
    token_labels = []
    masks = []
    for word, label in zip(words, word_labels):
        tmp_tokens = tokenizer.tokenize(word)
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
            print(tmp_tokens)
            tokens.append("[UNK]")
            token_labels.append(label)
            masks.append(1)
    return tokens, token_labels, masks


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_list.extend(["X", "[CLS]", "[SEP]", "[PAD]"])
    label_map = {label: i for i, label in enumerate(label_list)}
    print(label_map)

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        words = example.text_a.split()
        labels = example.label.split()

        tokens, labels, label_mask = sequence_labeling_tokenize(words, labels, tokenizer)

        # Account for [CLS] and [SEP] with "- 2"
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

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        if output_mode == "classification":
            label_ids = [label_map[x] for x in labels]
        elif output_mode == "regression":
            label_ids = float(example.label)
        else:
            raise KeyError(output_mode)

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

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("label: %s" % " ".join([str(x) for x in label_ids]))
            logger.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          label_ids=label_ids,
                          label_mask=label_mask,
                          attention_mask=attention_mask,
                          segment_ids=segment_ids))
    return features


def masked_simple_accuracy(preds, labels, masks):
    """masked label accuracy"""
    preds = np.reshape(preds, [-1])
    labels = np.reshape(labels, [-1])
    masks = np.reshape(masks, [-1]) == 1

    results = torch.masked_select(torch.tensor(1 * (preds == labels)).to("cuda"),
                                  torch.ByteTensor(masks).to("cuda")).cpu().numpy()
    return results.mean()


def masked_acc(preds, labels, masks):
    """accuracy"""
    acc = masked_simple_accuracy(preds, labels, masks)
    return {
        "acc": acc,
    }


def compute_metrics(task_name, preds, labels, masks, **kwargs):
    """map task and metrics"""
    assert len(preds) == len(labels)
    if task_name == "detect":

        out = {"acc": masked_simple_accuracy(preds, labels, masks)}
        out.update(f_measure(preds, labels, masks))
        # out.update()
        return out
    elif task_name == "cws":
        return masked_acc(preds, labels, masks)
    elif task_name in ("ner", "ner_v2"):

        out = {"acc": masked_simple_accuracy(preds, labels, masks)}
        out.update(mask_span_f1(preds, labels, masks, label_list=kwargs["label_list"]))
        return out
    else:
        raise KeyError(task_name)


PROCESSORS = {
    "detect": DetectProcessor,
    "cws": CwsProcessor,
    "ner": NerProcessor,
    "ner_v2": Nerv2Processor,
}

OUTPUT_MODES = {
    "detect": "classification",
    "cws": "classification",
    "ner": "classification",
    "ner_v2": "classification",
}
