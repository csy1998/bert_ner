# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fei_wang@shannonai.com
@Time       : 2019/8/9 15:55
@Description: 
"""
from typing import Tuple, List


class Tag(object):
    def __init__(self, term, tag, begin, end):
        self.term = term
        self.tag = tag
        self.begin = begin
        self.end = end

    def to_tuple(self):
        return tuple([self.term, self.begin, self.end])

    def __str__(self):
        return str({key: value for key, value in self.__dict__.items()})

    def __repr__(self):
        return str({key: value for key, value in self.__dict__.items()})


def bmes_decode(char_label_list: List[Tuple[str, str]]) -> Tuple[str, List[Tag]]:
    idx = 0
    length = len(char_label_list)
    tags = []
    while idx < length:
        term, label = char_label_list[idx]
        current_label = label[0]

        # correct labels
        if current_label in ["M", "E"]:
            current_label = "B"
        if idx + 1 == length and current_label == "B":
            current_label = "S"

        # merge chars
        if current_label == "O":
            idx += 1
            continue
        if current_label == "S":
            tags.append(Tag(term, label[2:], idx, idx + 1))
            idx += 1
            continue
        if current_label == "B":
            end = idx + 1
            while end + 1 < length and char_label_list[end][1][0] == "M":
                end += 1
            if char_label_list[end][1][0] == "E":  # end with E
                entity = "".join(char_label_list[i][0] for i in range(idx, end + 1))
                tags.append(Tag(entity, label[2:], idx, end + 1))
                idx = end + 1
            else:  # end with M/B
                entity = "".join(char_label_list[i][0] for i in range(idx, end))
                tags.append(Tag(entity, label[2:], idx, end))
                idx = end
            continue
        else:
            # print([label for term, label in char_label_list])
            raise Exception("不应该运行到这里")

    sentence = "".join(term for term, _ in char_label_list)
    return sentence, tags
