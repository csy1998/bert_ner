#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: Wei Wu
@license: Apache Licence
@file: shannon_loss.py
@time: 2019/10/01
@contact: wu.wei@pku.edu.cn
todo(yuxian): ohem & label smoothing
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module, CrossEntropyLoss
from typing import List


class CELoss(Module):
    """交叉熵损失函数"""
    def __init__(self, class_weight: List[int] = None):
        super().__init__()
        self.class_weight = torch.tensor(class_weight, dtype=torch.float32).cuda() if class_weight is not None else None
        self.ce_loss = CrossEntropyLoss(weight=self.class_weight, reduction='none')

    def forward(self, input, target, mask=None):
        """
        算loss
        :param input: (N, d1, C)
        :param target: (N, d1)
        :param mask: (N, d1)
        :return: scalar
        """
        transposed_input = torch.transpose(input, 1, -1)  # (N, C, d1)
        ce_loss = self.ce_loss(transposed_input, target)  # (N, d1)
        if mask is not None:
            return torch.sum(ce_loss * mask) / torch.sum(mask)
        else:
            return torch.mean(ce_loss)


class ClassBalanceLoss(CELoss):
    """类平衡损失函数，见https://arxiv.org/abs/1901.05555"""
    def __init__(self, samples_per_class: List[int] = None, beta=0.9999):
        super().__init__()
        effective_num = 1.0 - np.power(beta, samples_per_class)
        label_logits = (1.0 - beta) / np.array(effective_num)
        class_weight = label_logits / np.sum(label_logits) * len(samples_per_class)
        self.class_weight = torch.tensor(class_weight, dtype=torch.float32).cuda() if class_weight is not None else None
        self.ce_loss = CrossEntropyLoss(weight=self.class_weight, reduction='none')


class FocalLoss(Module):
    """focal loss，见https://arxiv.org/abs/1708.02002"""
    def __init__(self, gamma, class_weight: List[int] = None):
        super().__init__()
        self.gamma = gamma
        self.class_weight = torch.tensor(class_weight, dtype=torch.float32).cuda() if class_weight is not None else None
        self.ce_loss = CrossEntropyLoss(weight=self.class_weight, reduction='none')

    def forward(self, input, target, mask=None):
        """
        算loss
        :param input: (N, d1, C)
        :param target: (N, d1)
        :param mask: (N, d1)
        :return: scalar
        """
        transposed_input = torch.transpose(input, 1, -1)  # (N, C, d1)
        ce_loss = self.ce_loss(transposed_input, target)  # (N, d1)

        target_one_hot = (target.unsqueeze(-1) == torch.arange(input.shape[-1]).cuda()).float()
        selected_prob = torch.sum(F.softmax(input, dim=-1) * target_one_hot, dim=-1)  # (N, d1)
        multiplier = torch.pow(1 - selected_prob, self.gamma)  # (N, d1)
        weighted_loss = ce_loss * multiplier
        if mask is not None:
            return torch.sum(weighted_loss * mask) / torch.sum(mask)
        else:
            return torch.mean(weighted_loss)


class GHMC(Module):
    """GHM Classification Loss.
    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181
    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
    """
    def __init__(self, bins=10, momentum=0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float().cuda() / bins
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = torch.zeros(bins).cuda()

    def forward(self, input, target, mask=None):
        """Calculate the GHM-C loss.
        Args:
            input (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            mask (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        target_one_hot = (target.unsqueeze(-1) == torch.arange(input.shape[-1]).cuda()).float()
        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(input)
        else:
            mask = torch.ones_like(input)
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(input)

        # gradient length
        g = torch.abs(input.sigmoid().detach() - target_one_hot)

        valid = mask > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(input, target_one_hot, weights, reduction='sum') / tot
        return loss


class DiceLoss(Module):
    """Dice loss, see https://arxiv.org/abs/1606.04797"""

    def __init__(self, class_weight: List[int] = None, smooth: int = 1, exponent: int = 2):
        """
        构造函数
        :param class_weight: 每个类的权重
        :param smooth: 平滑系数
        :param exponent: 指数
        """
        super(DiceLoss, self).__init__()
        self.class_weight = class_weight
        self.smooth = smooth
        self.p = exponent

    def forward(self, input, target, mask=None):
        """
        算loss
        :param input: (N, d1, C)
        :param target: (N, d1)
        :param mask: (N, d1)
        :return: scalar
        """
        transformed_input = input.softmax(-1)  # (N, d1, C)
        transformed_target = (target.unsqueeze(-1) == torch.arange(input.shape[-1]).cuda()).float()  # (N, d1, C)
        if mask is not None:
            transformed_mask = mask.unsqueeze(-1).expand_as(input)
        else:
            transformed_mask = torch.ones_like(input)
        total_loss = 0

        for i in range(target.shape[1]):
            iflat = transformed_input.narrow(-1, i, 1).view(-1) * transformed_mask.narrow(-1, i, 1).view(-1)
            tflat = transformed_target.narrow(-1, i, 1).view(-1) * transformed_mask.narrow(-1, i, 1).view(-1)
            numerator = 2.0 * (iflat * tflat).mean() + self.smooth
            denominator = (iflat ** self.p).mean() + (tflat ** self.p).mean() + self.smooth
            dice_loss = 1 - numerator / denominator
            if self.class_weight is not None:
                dice_loss *= self.class_weight[i]
            total_loss += dice_loss

        return total_loss / target.shape[1]


class ShannonLoss(Module):
    """提供统一的接口"""
    def __init__(self, loss_type, **kwargs):
        super().__init__()
        if loss_type == 'cross_entropy':
            self.loss = CELoss(class_weight=kwargs.get('class_weight'))
        elif loss_type == 'class_balance':
            self.loss = ClassBalanceLoss(samples_per_class=kwargs['samples_per_class'], beta=kwargs['beta'])
        elif loss_type == 'focal':
            self.loss = FocalLoss(gamma=kwargs['gamma'], class_weight=kwargs.get('class_weight'))
        elif loss_type == 'ghmc':
            self.loss = GHMC(bins=kwargs['bins'], momentum=kwargs['momentum'])
        elif loss_type == 'dice':
            self.loss = DiceLoss(class_weight=kwargs['class_weight'], smooth=kwargs['smooth'], exponent=kwargs['exponent'])

    def forward(self, input, target, mask=None):
        """
        计算loss
        :param input:
        :param target:
        :param mask:
        :return:
        """
        return self.loss.forward(input, target, mask)


def main():
    """测试"""
    torch.random.manual_seed(42)

    # no_of_classes = 3
    # num_tokens = 2
    # num_batches = 1
    # B, L, C
    input = torch.tensor([[[2, 1, 3], [4, 6, 5]]], dtype=torch.float32).cuda()
    # B, L
    target = torch.tensor([[0, 1]], dtype=torch.long).cuda()
    mask = torch.tensor([[1, 0]], dtype=torch.float32).cuda()
    class_weight = [2, 3, 1]

    ce_loss = ShannonLoss('cross_entropy', class_weight=class_weight)
    ce_result = ce_loss.forward(input, target, mask)
    np.testing.assert_almost_equal(ce_result.cpu().data.tolist(), 2.8152, decimal=4)

    focal_loss = ShannonLoss('focal', gamma=2, class_weight=class_weight)
    focal_result = focal_loss.forward(input, target, mask)
    np.testing.assert_almost_equal(focal_result.cpu().data.tolist(), 1.6059, decimal=4)

    class_balance_loss = ShannonLoss('class_balance', samples_per_class=class_weight, beta=0.999)
    class_balance_result = class_balance_loss.forward(input, target, mask)
    np.testing.assert_almost_equal(class_balance_result.cpu().data.tolist(), 1.1519, decimal=4)

    ghmc_loss = ShannonLoss('ghmc', bins=10, momentum=0.1)
    ghmc_result = ghmc_loss.forward(input, target, mask)
    np.testing.assert_almost_equal(ghmc_result.cpu().data.tolist(), 1.6625, decimal=4)

    dice_loss = ShannonLoss('dice', class_weight=class_weight, smooth=1, exponent=2)
    dice_result = dice_loss.forward(input, target, mask)
    np.testing.assert_almost_equal(dice_result.cpu().data.tolist(), 0.1925, decimal=4)


if __name__ == '__main__':
    main()
