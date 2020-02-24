"""
@author: Shuyin Chen
@contact: shuyin_chen@shannonai.com

@version: 1.0
@file: set seed
@time: 2019/11/30 16:50
"""

import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from bert_ner.utils.logger import logger
from bert_ner.evaluators.evaluator_base import EvaluatorBase
from bert_ner.trainers.trainer_base import TrainerBase
from bert_ner.losses.loss import ShannonLoss


class EvaluatorForDistill(EvaluatorBase):
    """
    变量声明在 EvaluatorBase 中
    需要重载/重写的方法:
        init
        eval
    """
    def __init__(self,
                 params: dict,
                 model: nn.Module):
        logger.info('Initializing Evaluator')
        super(EvaluatorForDistill, self).__init__(params)

        self.model = model
        self.eval_dataloader = TrainerBase.load_dataset(params, prefix='dev', dataset_type="sequence_masking")
        self.t_total = len(self.eval_dataloader)

        if self.multi_gpu:
            self.model = TrainerBase.init_ddp(params, self.model)

    def eval(self):
        """
        eval model
        """
        if not self.is_master:
            return

        super().eval()
        self.model.eval()
        # if self.multi_gpu:
        #     torch.distributed.barrier()

        iter_bar = tqdm(self.eval_dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
        for batch in iter_bar:
            if self.params.n_gpu > 0:
                batch = tuple(t.to(f'cuda:{self.params.local_rank}') for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'masked_lm_labels': batch[2],
                      }
            with torch.no_grad():  # 加速和节约显存
                outputs = self.model(**inputs)
            loss = outputs[0]

            if self.multi_gpu:
                loss = loss.mean()
            assert loss.item() >= 0
            self.total_loss += loss.item()
            self.last_loss = loss.item()
            self.n_step += 1

            iter_bar.update()
            iter_bar.set_postfix({'loss_cur': f'{self.last_loss:.3f}',
                                  'loss': f'{self.total_loss / self.n_step:.3f}'})

        iter_bar.close()

        return self.total_loss / self.n_step


