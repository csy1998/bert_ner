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
from bert_ner.evaluators.evaluator_ner import EvaluatorForNER
from bert_ner.losses.loss import ShannonLoss


class TrainerForNER(TrainerBase):
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
        super(TrainerForNER, self).__init__(params)

        self.train_dataloader = self.load_dataset(params, prefix='train', dataset_type="sequence_labeling")
        self.model, self.tokenizer = modules
        self.optimizer = self.init_optimizer([self.model])
        self.loss_fn = ShannonLoss(self.params.loss_type)

        if self.fp16:
            self.model, self.optimizer = self.init_fp16([self.model, self.optimizer])

        if self.multi_gpu:
            self.model = self.init_ddp(params, self.model)

        if self.do_eval:
            self.eval_loss = 0
            self.evaluator = EvaluatorForNER(params, self.model)

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
                          'labels': batch[1],
                          'label_mask': batch[2],
                          'token_type_ids': batch[4],
                          'attention_mask': batch[3],
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
        One optimization step:
        Input:
            input_ids: `torch.tensor(bs, seq_length)` - The token ids.
            labels: `torch.tensor(bs, seq_length)`
            label_mask: `torch.tensor(bs, seq_length)`
            attention_mask: `torch.tensor(bs, seq_length)` - The attention mask for self attention.
            token_type_ids: `torch.tensor(bs, seq_length)`
        """
        assert inputs
        outputs = self.model(**inputs)
        logits = outputs[0]
        labels = inputs['labels']
        mask = inputs['label_mask']

        # todo 注意 若是bert_crf_tagger 那么 loss 需要在 model 内部计算
        loss = self.loss(logits, labels, mask)
        assert loss.item() >= 0

        self.optimize(loss)
        self.end_step()

    def loss(self, logits=None, labels=None, mask=None):
        """
        calculate loss using self.loss
        """
        if mask is not None:
            loss_mask = (mask == 1).float()
        else:
            loss_mask = None
        loss = self.loss_fn(logits, labels, loss_mask)
        return loss

    def eval_checkpoint(self):
        """
        evaluate model when
            1. n_step_global % checkpoint_interval == 0
            2. each epoch ends
        """
        if self.is_master and self.do_eval:

            self.eval_loss, results = self.evaluator.eval()
            self.eval_result_cur = results['span-f1']

            print("eval_span_f1: ", self.eval_result_cur)
            print("eval_span_f1_best: ", self.eval_result_best)

            self.end_eval_checkpoint()

    def save_models(self, path):
        """
        save pretrained model
        """
        model = self.model
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # save label map
        label_map_path = os.path.join(path, 'label_map.txt')
        with open(label_map_path, 'w') as label_map_f:
            for label in self.evaluator.label_list:
                label_map_f.write(label + '\n')

    def save_eval_result(self):
        """
        Save eval results
        """
        with open(self.eval_result_path, 'a') as eval_result_f:
            result = f"epoch: {self.epoch} \t"
            result += f"loss: {self.eval_loss} \t"
            result += f"span f1: {self.eval_result_cur}\t"
            result += f"best={self.best_k} \t"
            eval_result_f.write(result + '\n')

            if self.epoch == self.params.n_epoch - 1:
                result = f"best epoch: {self.best_k} \t"
                result += f"best span f1: {self.eval_result_best} \t"
                eval_result_f.write('\n' + result + '\n')

