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
from bert_ner.losses.distill_loss import HiddenStateMSELossForMobileBert


class TrainerForMobileBert(TrainerBase):
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
        super(TrainerForMobileBert, self).__init__(params)

        self.stage = None
        self.num_stages = 13
        self.pkt_epochs = params.pkt_epochs
        self.pd_epochs = params.pd_epochs
        self.epochs_for_stage = [self.pkt_epochs] * (self.num_stages - 1) + [self.pd_epochs]

        self.train_dataloader = self.load_dataset(params, prefix='train', dataset_type="sequence_masking")
        self.student, self.teacher, self.tokenizer = modules
        self.loss_hidn_fn = HiddenStateMSELossForMobileBert(student_hidden_size=self.student.config.hidden_size,
                                                            teacher_hidden_size=self.teacher.config.hidden_size)

        if self.do_eval:
            self.eval_result_best = 1e8
            self.evaluator = EvaluatorForDistill(params, self.student)

            # 保存 progressive knowledge distillation 之后各层都得到初始化的 mobilebert
            self.init_model_path = os.path.join(self.output_dir, 'mobilebert_init')
            self.make_dirs(self.init_model_path)

    def train(self):
        """
        The real training loop.
        """
        stage_start = 12 if self.params.distill else 0
        for stage in range(stage_start, self.num_stages):
            # 去掉 module.
            while isinstance(self.student, torch.nn.parallel.DistributedDataParallel):
                self.student = self.student.module

            self.student.float()
            self.teacher.float()

            # freeze other params except layer stage
            self.stage = stage
            if self.stage <= 11:
                for name, param in self.student.named_parameters():
                    if 'layer.' + str(self.stage) + '.' in name:
                        param.requires_grad = True
                    else:
                        # param = param.float()
                        param.requires_grad = False
            else:
                for name, param in self.student.named_parameters():
                    param.requires_grad = True

            # print("all param")
            # for name, param in student_model.named_parameters():
            #     print(name, param.requires_grad)

            self.params.n_epoch = self.epochs_for_stage[self.stage]
            self.optimizer = self.init_optimizer([self.student])

            if self.fp16:
                self.student, self.teacher, self.optimizer = self.init_fp16(
                    [self.student, self.teacher, self.optimizer])

            if self.multi_gpu:
                self.student = self.init_ddp(self.student)

            super().train()
            self.student.train()
            self.teacher.eval()
            self.student.zero_grad()
            stage_input = self.stage if self.stage <= 11 else None

            for _ in range(self.params.n_epoch):
                if self.is_master: logger.info(f'--- Starting epoch {self.epoch} / {self.params.n_epoch - 1}')
                # if self.multi_gpu:
                #     torch.distributed.barrier()

                iter_bar = tqdm(self.train_dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
                for batch in iter_bar:
                    if self.params.n_gpu > 0:
                        batch = tuple(t.to(f'cuda:{self.params.local_rank}') for t in batch)
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'masked_lm_labels': batch[2],
                              'stage': stage_input,
                              }
                    self.step(inputs)

                    iter_bar.update()
                    current_lr = self.scheduler.get_lr()[0]
                    iter_bar.set_postfix({'lr': f'{current_lr:.5f}',
                                          'loss_cur': f'{self.last_loss:.3f}',
                                          'loss_glo': f'{self.total_loss_global * self.params.gradient_accumulation_steps / self.n_step_global:.3f}'})
                iter_bar.close()

                if self.is_master:
                    logger.info(f'--- Ending epoch {self.epoch} / {self.params.n_epoch - 1}')
                self.end_epoch()

                if self.patience > self.early_stop_patience > 0:
                    print("training stopped because of early stopping!!!")
                    break

            self.epoch = 0
            if self.stage < 12:
                self.save_models(self.init_model_path)

    def step(self, inputs=None):
        """
        One optimization step
        Input:
            input_ids: `torch.tensor(bs, seq_length)`
            input_mask: `torch.tensor(bs, seq_length)`
            lm_label_ids: `torch.tensor(bs, seq_length)`
        """
        attention_mask = inputs['attention_mask']
        masked_lm_labels = inputs['masked_lm_labels']

        student_outputs = self.student(**inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)

        loss = self.loss(student_outputs, teacher_outputs, attention_mask, masked_lm_labels)
        assert loss.item() >= 0

        self.optimize(loss)
        self.end_step()

    def loss(self, student_outputs=None, teacher_outputs=None, attention_mask=None, masked_lm_labels=None):
        """
        calculate loss
        student_outputs:
            loss
            logits: B * L * V
            hidden_states: List[ B * L * hidden_size(768) ]
        """
        loss = None
        if self.stage < 12:
            # hidden_states
            student_hidden_states = student_outputs[2]
            teacher_hidden_states = teacher_outputs[2]
            active_hidden_states = attention_mask.view(-1) == 1

            active_student_hidden_states = student_hidden_states[-1].view(-1, self.student.config.hidden_size)[
                active_hidden_states]
            active_teacher_hidden_states = teacher_hidden_states[-1].view(-1, self.teacher.config.hidden_size)[
                active_hidden_states]

            assert active_student_hidden_states.requires_grad and not active_teacher_hidden_states.requires_grad

            loss_hidn_list = self.loss_hidn_fn(active_student_hidden_states,
                                               active_teacher_hidden_states)

            loss_hidn, loss_hidn_mean, loss_hidn_var = loss_hidn_list

            loss = self.params.alpha_hidn * loss_hidn + \
                   self.params.alpha_hidn_mean * loss_hidn_mean + \
                   self.params.alpha_hidn_var * loss_hidn_var

        if self.stage == 12:
            # logits
            student_logits = student_outputs[1]
            teacher_logits = teacher_outputs[1]
            active_logits = masked_lm_labels.view(-1) != -1
            active_student_logits = student_logits.view(-1, self.tokenizer.vocab_size)[active_logits]
            active_teacher_logits = teacher_logits.view(-1, self.tokenizer.vocab_size)[active_logits]

            loss_mlm = student_outputs[0]
            loss_pred = (- F.softmax(active_teacher_logits, dim=-1) * F.log_softmax(
                active_student_logits / self.params.temperature, dim=-1)).sum(-1).mean()
            loss = self.params.alpha_pred * loss_pred + self.params.alpha_mlm * loss_mlm

        return loss

    def end_epoch(self):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        Do some tensorboard logging and checkpoint saving.
        """
        if self.stage == 12:
            self.eval_checkpoint()
        self.epoch += 1

    def eval_checkpoint(self):
        """
        evaluate model when
            1. n_step_global % checkpoint_interval == 0
            2. each epoch ends
        """
        if self.stage > 11 and self.is_master and self.do_eval:
            self.eval_result_cur = self.evaluator.eval()

            print("eval_loss: ", self.eval_result_cur)
            print("eval_loss_best: ", self.eval_result_best)

            self.end_eval_checkpoint(mode=-1)

    def save_models(self, path):
        """
        save pretrained model
        """
        model = self.student
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
