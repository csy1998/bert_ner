# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: test_span_f1
@time: 2019/11/19 16:38

    这一行开始写关于本文件的说明与解释
"""

from typing import List, Dict
from bert_ner.metrics.span_f1 import mask_span_f1
import numpy as np


def test_span_f1():
    """test span f1"""
    batch_preds = [[0, 1, 2, 0, 0]]
    batch_labels = [[0, 1, 2, 0, 3]]
    batch_masks = None
    label_list: List[str] = ["O", "B-I", "E-I", "S-P"]

    # all tags
    span_f1 = mask_span_f1(batch_preds=batch_preds, batch_labels=batch_labels,
                           batch_masks=batch_masks, label_list=label_list)

    golden = {'span-precision': 1.0,
              'span-recall ': 0.5,
              'span-f1': 2/3}

    assert dict_equal(span_f1, golden)

    # specific tags
    span_f1 = mask_span_f1(batch_preds=batch_preds, batch_labels=batch_labels,
                           batch_masks=batch_masks, label_list=label_list,
                           specific_tags=["I"])
    golden = {'span-precision': 1.0,
              'span-recall ': 1.0,
              'span-f1': 1.0}
    assert dict_equal(span_f1, golden)


def dict_equal(dic1: Dict, dic2: Dict) -> bool:
    """compare two dicts"""
    dic1_keys = set(dic1.keys())
    dic2_keys = set(dic2.keys())
    if dic1_keys != dic2_keys:
        return False
    dic1_lst = []
    dic2_lst = []
    for key in dic1_keys:
        dic1_lst.append(dic1[key])
        dic2_lst.append(dic2[key])
    if np.allclose(np.array(dic1_lst), np.array(dic2_lst)):
        return True
    return False
