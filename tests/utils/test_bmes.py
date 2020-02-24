# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: test_bmes
@time: 2019/11/19 14:54

    这一行开始写关于本文件的说明与解释
"""

from bert_ner.utils.bmes_decode import bmes_decode


def test_bmes_decode():
    """test bmes decoding"""

    x = [("我", "S"), ("爱", "O"), ("北", "B-LOC"), ("京", "M-LOC"), ("天", "B-ARC"), ("安", "M-ARC"), ("门", "E-ARC")]
    golden = [('我', 0, 1), ('北京', 2, 4), ('天安门', 4, 7)]
    decode_sent, decode_tags = bmes_decode(x)
    assert [t.to_tuple() for t in decode_tags] == golden
