"""
@author: Shuyin Chen
@contact: shuyin_chen@shannonai.com

@version: 1.0
@file: set seed
@time: 2019/11/30 14:50
"""

import os
import math
import numpy as np
from tqdm import tqdm
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_ner.utils.logger import logger
from bert_ner.trainers.trainer_base import TrainerBase
from bert_ner.evaluators.evaluator_distill import EvaluatorForDistill


class TrainerForFineTuneBertLarge(TrainerBase):
    """
    大部分变量和方法声明在 TrainerBase 中
    需要重载/重写的方法:
        init
        train
        step
        loss
        eval_checkpoint
        save_models
        save_eval_result
    """
    def __init__(self,
                 params: dict,
                 modules: List[nn.Module]):
        super(TrainerForFineTuneBertLarge, self).__init__(params)

        self.train_dataloader = self.load_dataset(params, prefix='train', dataset_type="sequence_masking")
        self.model, self.tokenizer = modules
        self.optimizer = self.init_optimizer([self.model])

        if self.fp16:
            self.model, self.optimizer = self.init_fp16([self.model, self.optimizer])

        if self.multi_gpu:
            self.model = self.init_ddp(params, self.model)

        if self.do_eval:
            self.eval_result_best = 1e8
            self.evaluator = EvaluatorForDistill(params, self.model)

    def train(self):
        """
        The real training loop.
        """
        super().train()

        self.model.train()
        self.model.zero_grad()

        for _ in range(self.params.n_epoch):
            if self.is_master: logger.info(f'--- Starting epoch {self.epoch}/{self.params.n_epoch-1}')
            # if self.multi_gpu:
            #     torch.distributed.barrier()

            iter_bar = tqdm(self.train_dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
            for batch in iter_bar:
                if self.params.n_gpu > 0:
                    batch = tuple(t.to(f'cuda:{self.params.local_rank}') for t in batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'masked_lm_labels': batch[2],
                          }
                self.step(inputs)

                iter_bar.update()
                current_lr = self.scheduler.get_lr()[0]
                iter_bar.set_postfix({'lr': f'{current_lr:.5f}',
                                      'loss_cur': f'{self.last_loss:.3f}',
                                      'loss_glo': f'{self.total_loss_global * self.params.gradient_accumulation_steps / self.n_step_global:.3f}'})
            iter_bar.close()

            if self.is_master:
                logger.info(f'--- Ending epoch {self.epoch} / {self.params.n_epoch-1}')
            self.end_epoch()

            if self.patience > self.early_stop_patience > 0:
                print("training stopped because of early stopping!!!")
                break

    def step(self, inputs=None):
        """
        One optimization step
        Input:
            input_ids: `torch.tensor(bs, seq_length)`
            input_mask: `torch.tensor(bs, seq_length)`
            lm_label_ids: `torch.tensor(bs, seq_length)`
        """
        assert inputs
        outputs = self.model(**inputs)

        loss = outputs[0]
        assert loss.item() >= 0

        self.optimize(loss)
        self.end_step()

    def loss(self):
        """
        calculate loss
        """
        return None

    def eval_checkpoint(self):
        """
        evaluate model when
            1. n_step_global % checkpoint_interval == 0
            2. each epoch ends
        """
        if self.is_master and self.do_eval:
            # with torch.no_grad:
            self.eval_result_cur = self.evaluator.eval()

            print("eval_loss: ", self.eval_result_cur)
            print("eval_loss_best: ", self.eval_result_best)

            self.end_eval_checkpoint(mode=-1)

    def save_models(self, path):
        """
        save pretrained model
        """
        model = self.model
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def save_eval_result(self):
        """
        Save eval results
        """
        with open(self.eval_result_path, 'a') as eval_result_f:
            result = f"epoch: {self.epoch} \t"
            result += f"loss: {self.eval_result_cur} \t"
            result += f"best={self.best_k} \t"
            eval_result_f.write(result + '\n')

            if self.epoch == self.params.n_epoch - 1:
                result = f"best epoch: {self.best_k} \t"
                result += f"best loss: {self.eval_result_best} \t"
                eval_result_f.write('\n' + result + '\n')

