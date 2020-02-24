# encoding: utf-8
"""
@author: Shuyin Chen
@contact: shuyin_chen@shannonai.com

@version: 1.0
@file: entity classifier
@time: 2019/12/16 21:20

    这一行开始写关于本文件的说明与解释
"""

import math
import torch
from torch import nn


class SingleLinearClassifier(nn.Module):
    """
    SingleLinearClassifier
    """
    def __init__(self, hidden_size, num_label):
        super(SingleLinearClassifier, self).__init__()
        self.num_label = num_label
        self.classifier = nn.Linear(hidden_size, num_label)

    def forward(self, input_features):
        features_output = self.classifier(input_features)
        return features_output


class MultiNonLinearClassifier(nn.Module):
    """
    MultiNonLinearClassifier
    """
    def __init__(self, hidden_size, num_label):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.classifier1 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.classifier2 = nn.Linear(int(hidden_size / 2), num_label)

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        features_output1 = nn.ReLU()(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}


class FeedForwardNet(nn.Module):
    """
    FFN in bert
    """
    def __init__(self, hidden_size, intermediate_size, output_size, act_fn='relu'):
        super(FeedForwardNet, self).__init__()
        self.dense1 = nn.Linear(int(hidden_size), int(intermediate_size))
        self.intermediate_act_fn = ACT2FN[act_fn]
        self.dense2 = nn.Linear(int(intermediate_size), int(output_size))

    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states


class BiaffineClassifier(nn.Module):
    """
    Biaffine Classifier
    """
    def __init__(self, hidden_size, num_label):
        super(BiaffineClassifier, self).__init__()
        self.num_label = num_label
        self.dense = nn.Linear(int(hidden_size*2), self.num_label)
        self.u = torch.randn(num_label, hidden_size, hidden_size, requires_grad=True)

    def forward(self, hidden_states1, hidden_states2):
        """
        :param hidden_states1: L * H
        :param hidden_states2: L * H
        :return: L * N
        """
        hidden_states = self.dense(torch.cat([hidden_states1, hidden_states2], -1))     # L * N

        hidden_states_prev = torch.matmul(hidden_states1, self.u)   # N * L * H
        shape = hidden_states_prev.shape
        hidden_states_prev.reshape(shape[1], shape[0], -1)          # L * N * H
        hidden_states_prev = torch.matmul(hidden_states_prev, hidden_states2.unsqueeze(-1))
        hidden_states += hidden_states_prev.squeeze(-1)
        return hidden_states


class NERLabelClassifier(nn.Module):
    """
    NER Label Classifier
    output: score before and after softmax
    """
    def __init__(self, hidden_size, num_label):
        super(NERLabelClassifier, self).__init__()
        self.num_label = num_label
        intermediate_size = hidden_size / 2
        self.feed_forward = FeedForwardNet(hidden_size, intermediate_size, output_size=self.num_label)

    def forward(self, hidden_states):
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states


class EntityClassifier(nn.Module):
    """
    Entity Classifier
    output: score before and after softmax
    """
    def __init__(self, hidden_size, biaffine_size, num_label, label_embedding_size=128):
        # hidden_size -> intermediate_size -> biaffine_size -> num_label_size
        super(EntityClassifier, self).__init__()
        self.num_label = num_label
        self.label_embedding = torch.randn(label_embedding_size, 11, requires_grad=True)

        intermediate_size = hidden_size * 4
        self.head_ffn = FeedForwardNet(hidden_size, intermediate_size, output_size=biaffine_size)
        self.tail_ffn = FeedForwardNet(hidden_size, intermediate_size, output_size=biaffine_size)
        self.biaffine_classifier = BiaffineClassifier(biaffine_size, self.num_label)

    def forward(self, head_hidden_states, tail_hidden_states, label_head, label_tail, cat_label=True):

        # concat label embedding
        if cat_label:
            head_hidden_states = torch.cat([head_hidden_states, self.label_embedding[label_head]], -1)
            tail_hidden_states = torch.cat([tail_hidden_states, self.label_embedding[label_tail]], -1)

        # feed forward
        head_hidden_states = self.head_ffn(head_hidden_states)
        tail_hidden_states = self.tail_ffn(tail_hidden_states)

        # classify
        hidden_states = self.biaffine_classifier(head_hidden_states, tail_hidden_states)
        return hidden_states


