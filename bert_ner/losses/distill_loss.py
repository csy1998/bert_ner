# -*- coding: utf-8 -*-
"""
@Author     : Shuyin Chen
@Contact    : shuyin_chen@shannonai.com
@Time       : 2019/9/20 13:00
@Description: focal loss 
"""

import torch
import torch.nn as nn


class EmbeddingMSELoss(nn.Module):
    """
    EmbeddingMSELoss
    """
    def __init__(self, student_hidden_size=None, teacher_hidden_size=None, size_average=True):
        super(EmbeddingMSELoss, self).__init__()
        self.student_hidden_size = student_hidden_size
        self.teacher_hidden_size = teacher_hidden_size
        self.embedding_transform = nn.Linear(student_hidden_size, teacher_hidden_size)
        self.loss_mse_fn = nn.MSELoss()
        self.size_average = size_average

    def forward(self, student_embedding, teacher_embedding):
        """
        :param student_embedding: B * L * student_hidden_size
        :param teacher_embedding: B * L * teacher_hidden_size
        :return: embedding_loss
        """
        if self.student_hidden_size != self.teacher_hidden_size:
            student_embedding = self.embedding_transform(student_embedding)

        loss = self.loss_mse_fn(student_embedding.float(), teacher_embedding.float())

        return loss


class HiddenStateMSELoss(nn.Module):
    """
    HiddenStateMSELoss
    """
    def __init__(self, student_hidden_size=None, teacher_hidden_size=None, size_average=True):
        super(HiddenStateMSELoss, self).__init__()
        self.student_hidden_size = student_hidden_size
        self.teacher_hidden_size = teacher_hidden_size
        self.hidden_state_transform = nn.Linear(student_hidden_size, teacher_hidden_size)
        self.loss_mse_fn = nn.MSELoss()
        self.size_average = size_average

    def forward(self, student_hidden_state, teacher_hidden_state):
        """
        student_embedding: B * L * student_hidden_size
        teacher_embedding: B * L * teacher_hidden_size
        """
        if self.student_hidden_size != self.teacher_hidden_size:
            student_hidden_state = self.hidden_state_transform(student_hidden_state)

        loss = self.loss_mse_fn(student_hidden_state.float(), teacher_hidden_state.float())

        return loss


class HiddenStateMSELossForMobileBert(nn.Module):
    """
    HiddenStateMSELossForMobileBert
    """
    def __init__(self, student_hidden_size=None, teacher_hidden_size=None, size_average=True):
        super(HiddenStateMSELossForMobileBert, self).__init__()
        self.student_hidden_size = student_hidden_size
        self.teacher_hidden_size = teacher_hidden_size
        self.hidden_state_transform = nn.Linear(student_hidden_size, teacher_hidden_size)
        self.loss_mse_fn = nn.MSELoss()
        self.size_average = size_average
        self.layerNorm = nn.LayerNorm(teacher_hidden_size, elementwise_affine=False)

    def forward(self, student_hidden_state, teacher_hidden_state):
        """
        student_embedding: B * L * student_hidden_size
        teacher_embedding: B * L * teacher_hidden_size
        """
        if self.student_hidden_size != self.teacher_hidden_size:
            student_hidden_state = self.hidden_state_transform(student_hidden_state)

        norm_student_hidden_state = self.layerNorm(student_hidden_state)
        norm_teacher_hidden_state = self.layerNorm(teacher_hidden_state)
        hidden_loss = self.loss_mse_fn(norm_student_hidden_state.float(), norm_teacher_hidden_state.float())

        mean_s, mean_t = student_hidden_state.mean(dim=-1), teacher_hidden_state.mean(dim=-1)
        hidden_mean_loss = self.loss_mse_fn(mean_s.float(), mean_t.float())

        var_s, var_t = student_hidden_state.var(dim=-1), teacher_hidden_state.var(dim=-1)
        hidden_var_loss = (abs(mean_s.float() - mean_t.float())).mean()

        return [hidden_loss, hidden_mean_loss, hidden_var_loss]