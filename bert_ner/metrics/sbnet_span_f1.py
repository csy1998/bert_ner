# -*- coding: utf-8 -*-
"""
@Author     : Shuyin Chen
@Contact    : shuyin_chen@shannonai.com
@Time       : 2019/8/9 15:55
@Description:
"""

from typing import List, Set, Tuple


class Tag(object):
    """ Entity Tag """
    def __init__(self, instance_id, begin, end, type=None):
        self.instance_id = instance_id
        self.begin = begin
        self.end = end
        self.type = type

    def to_str(self):
        return str(self.instance_id) + ' ' + str(self.begin) + ' ' + str(self.end)


def sbnet_decode(instance_id, label_list: List[str]):
    """
    decode for stack-based ner
    """
    idx = 0
    tags = []
    length = len(label_list)

    while idx < length:

        while idx < length and label_list[idx] == 'O':
            idx += 1
        start = idx

        while idx < length and label_list[idx] != 'O':
            idx += 1
        end = idx

        if start >= length or end >= length:
            break
        if start >= end:
            print("label_list: ", label_list)
            print("start: ", start)
            print("end: ", end)
        assert start < end
        stack = []
        for i in range(start, end):
            if '(' in label_list[i]:
                stack += [i] * label_list[i].count('(')
            if ')' in label_list[i] and len(stack) != 0:
                entity_start = stack.pop()
                entity_end = i
                tags.append(Tag(instance_id, entity_start, entity_end+1))

    return tags


def trunc_by_mask(lst: List, masks: List) -> List:
    """根据mask truncate lst"""
    out = []
    for item, mask in zip(lst, masks):
        if mask:
            out.append(item)
    return out


def get_entity(batch_preds, batch_masks=None, label_list=None, instance_cnt=None):
    """
    get_entity
    """
    total_tags = []
    for instance_id, (preds, masks) in enumerate(zip(batch_preds, batch_masks)):
        if masks is not None:
            preds = trunc_by_mask(preds, masks)
        preds = [label_list[idx] if idx < len(label_list) else "O" for idx in preds]
        pred_tags: List[Tag] = sbnet_decode(instance_id + instance_cnt, preds)
        total_tags += pred_tags
    return total_tags


def sbnet_span_f1(batch_preds, batch_labels, batch_masks=None, label_list=None):
    """
    计算span-based F1
    Args:
        batch_preds: 模型的预测. [batch, length]
        batch_labels: ground truth. [batch, length]
        label_list: label_list[idx] = label_idx。每一位是一个label
        batch_masks: [batch, length]

    Returns:
        span-based f1
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_tags = []

    if batch_masks is None:
        batch_masks = [None] * len(batch_preds)

    for instance_id, (preds, labels, masks) in enumerate(zip(batch_preds, batch_labels, batch_masks)):
        if masks is not None:
            preds = trunc_by_mask(preds, masks)
            labels = trunc_by_mask(labels, masks)

        preds = [label_list[idx] if idx < len(label_list) else "O" for idx in preds]
        labels = [label_list[idx] for idx in labels]

        pred_tags: List[Tag] = sbnet_decode(instance_id, preds)
        golden_tags: List[Tag] = sbnet_decode(instance_id, labels)

        total_tags += pred_tags

        pred_set: Set[Tuple] = set((tag.begin, tag.end) for tag in pred_tags)
        golden_set: Set[Tuple] = set((tag.begin, tag.end) for tag in golden_tags)

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

    results = {
        "span-precision": precision,
        "span-recall ": recall,
        "span-f1": f1
    }
    return results, total_tags


def sbnet_f1(pred_tags, golden_tags, pred_types, golden_types, type_list):
    """
    计算 F1
    Args:
        pred_tags: List[Tag]
        golden_tags: List[Tag]

    Returns:
        ner / cls/ total F1 score
    """
    results = {}

    # ner F1 score
    ner_true_pos = 0
    ner_false_pos = 0
    ner_false_neg = 0

    ner_pred_set: Set[Tuple] = set((tag.instance_id, tag.begin, tag.end) for tag in pred_tags)
    ner_golden_set: Set[Tuple] = set((tag.instance_id, tag.begin, tag.end) for tag in golden_tags)

    for pred in ner_pred_set:
        if pred in ner_golden_set:
            ner_true_pos += 1
        else:
            ner_false_pos += 1

    for pred in ner_golden_set:
        if pred not in ner_pred_set:
            ner_false_neg += 1

    ner_precision = ner_true_pos / (ner_true_pos + ner_false_pos + 1e-10)
    ner_recall = ner_true_pos / (ner_true_pos + ner_false_neg + 1e-10)
    ner_f1 = 2 * ner_precision * ner_recall / (ner_precision + ner_recall + 1e-10)

    ner_results = {'precision': ner_precision, 'recall': ner_recall, 'f1': ner_f1}
    results['ner'] = ner_results


    # cls F1 score
    cls_acc = 0
    cnt = [1 if pred_types[i] == golden_types[i] else 0 for i in range(len(pred_types))]
    cls_acc = sum(cnt) / len(pred_types)
    results['cls'] = cls_acc

    # total F1 score
    true_pos = 0
    false_pos = 0
    false_neg = 0

    pred_set: Set[Tuple] = set((tag.instance_id, tag.begin, tag.end, tag.type) for tag in pred_tags if tag.type < len(type_list)-1)
    golden_set: Set[Tuple] = set((tag.instance_id, tag.begin, tag.end, tag.type) for tag in golden_tags)

    for pred in pred_set:
        if pred in golden_set:
            true_pos += 1
        else:
            false_pos += 1

    for pred in golden_set:
        if pred not in pred_set:
            false_neg += 1

    precision = true_pos / (true_pos + false_pos + 1e-10)
    recall = true_pos / (true_pos + false_neg + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    total_results = {'precision': precision, 'recall': recall, 'f1': f1}
    results['total'] = total_results

    return results

