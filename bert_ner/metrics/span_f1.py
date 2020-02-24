# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: span_f1
@time: 2019/11/12 11:06

    这一行开始写关于本文件的说明与解释
"""

from typing import List, Set, Tuple
from bert_ner.utils.bmes_decode import bmes_decode, Tag


def mask_span_f1(batch_preds, batch_labels, batch_masks=None, label_list: List[str] = None,
                 specific_tags: List[str] = None):
    """
    计算span-based F1
    Args:
        batch_preds: 模型的预测. [batch, length]
        batch_labels: ground truth. [batch, length]
        label_list: label_list[idx] = label_idx。每一位是一个label
        batch_masks: [batch, length]
        specific_tags

    Returns:
        span-based f1
    """
    fake_term = "一"
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    if batch_masks is None:
        batch_masks = [None] * len(batch_preds)

    for preds, labels, masks in zip(batch_preds, batch_labels, batch_masks):
        if masks is not None:
            preds = trunc_by_mask(preds, masks)
            labels = trunc_by_mask(labels, masks)

        # print(masks)
        # print(preds)
        # print(labels)
        # print(label_list)

        preds = [label_list[idx] if idx < len(label_list) else "O" for idx in preds]
        labels = [label_list[idx] for idx in labels]

        pred_tags: List[Tag] = bmes_decode(char_label_list=[(fake_term, pred) for pred in preds])[1]
        golden_tags: List[Tag] = bmes_decode(char_label_list=[(fake_term, label) for label in labels])[1]

        # print("preds: ")
        # print(pred_tags)
        # print("golden: ")
        # print(golden_tags)

        if specific_tags is not None:
            pred_set: Set[Tuple] = set((tag.begin, tag.end, tag.tag) for tag in pred_tags if tag.tag in specific_tags)
            golden_set: Set[Tuple] = set((tag.begin, tag.end, tag.tag) for tag in golden_tags if tag.tag in specific_tags)
        else:
            pred_set: Set[Tuple] = set((tag.begin, tag.end, tag.tag) for tag in pred_tags)
            golden_set: Set[Tuple] = set((tag.begin, tag.end, tag.tag) for tag in golden_tags)

        for pred in pred_set:
            if pred in golden_set:
                true_positives += 1
            else:
                false_positives += 1

        for pred in golden_set:
            if pred not in pred_set:
                false_negatives += 1

    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return {
        "span-precision": precision,
        "span-recall ": recall,
        "span-f1": f1
    }


def trunc_by_mask(lst: List, masks: List) -> List:
    """根据mask truncate lst"""
    out = []
    for item, mask in zip(lst, masks):
        if mask:
            out.append(item)
    return out
