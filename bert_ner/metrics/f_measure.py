# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: metrics
@time: 2019/11/1 19:22

    定义一些metrics
"""


def masked_recall(preds, labels, masks=None, pos_idx=1):
    """计算recall"""
    pred1 = preds == pos_idx
    label1 = labels == pos_idx
    true_positives = (pred1 * label1).sum()
    golden_positives = label1.sum()

    return true_positives/(golden_positives+1e-9)


def masked_precision(preds, labels, masks=None, pos_idx=1):
    """计算precision"""
    pred1 = preds == pos_idx
    label1 = labels == pos_idx
    # print(pred1)
    # print(label1)
    true_positives = (pred1 * label1).sum()
    pred_positives = (pred1 * masks).sum()

    return true_positives / (pred_positives+1e-9)


def f_measure(preds, labels, masks, pos_idx=1):
    """f1 score"""
    recall = masked_recall(preds, labels, masks, pos_idx=pos_idx)
    precision = masked_precision(preds, labels, masks, pos_idx=pos_idx)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1
    }
