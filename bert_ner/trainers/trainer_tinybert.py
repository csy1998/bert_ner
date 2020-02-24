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
from bert_ner.losses.distill_loss import EmbeddingMSELoss, HiddenStateMSELoss


class TrainerForTinyBert(TrainerBase):
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
        super(TrainerForTinyBert, self).__init__(params)

        self.train_dataloader = self.load_dataset(params, prefix='train', dataset_type="sequence_masking")
        self.student, self.teacher, self.tokenizer = modules
        self.optimizer = self.init_optimizer([self.student])

        self.distill_layer_idx = [2, 5, 8, 11]
        self.loss_mse_fn = torch.nn.MSELoss()
        self.loss_embd_fn = EmbeddingMSELoss(student_hidden_size=self.student.config.hidden_size,
                                             teacher_hidden_size=self.teacher.config.hidden_size)
        self.loss_hidn_fn = HiddenStateMSELoss(student_hidden_size=self.student.config.hidden_size,
                                               teacher_hidden_size=self.teacher.config.hidden_size)

        if self.fp16:
            self.student, self.teacher, self.optimizer = self.init_fp16([self.student, self.teacher, self.optimizer])

        if self.multi_gpu:
            self.student = self.init_ddp(params, self.student)

        if self.do_eval:
            self.eval_result_best = 1e8
            self.evaluator = EvaluatorForDistill(params, self.student)

    def train(self):
        """
        The real training loop.
        """
        super().train()
        self.student.train()
        self.teacher.eval()
        self.student.zero_grad()

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
        output:
            loss
            logits: B * L * V
            hidden_states: List[ B * L * hidden_size(768) ]
            last_hidden_states: B * L * hidden_size(768)
            attention scores: List[ B * H * L * L ]
            embedding: B * L * hidden_size(768)
        """
        # embedding
        student_embedding = student_outputs[-1]
        teacher_embedding = teacher_outputs[-1]
        active_embedding = attention_mask.view(-1) == 1
        active_student_embedding = student_embedding.view(-1, self.student.config.hidden_size)[active_embedding]
        active_teacher_embedding = teacher_embedding.view(-1, self.teacher.config.hidden_size)[active_embedding]

        # hidden_states
        student_hidden_states = student_outputs[2]  # List[ B * L * hidden_size(768) ]
        teacher_hidden_states = teacher_outputs[2]  # List[ B * L * hidden_size(768) ]
        teacher_hidden_states = [teacher_hidden_states[idx] for idx in
                                 self.distill_layer_idx]  # List[ B * L * hidden_size(768) ]
        active_hidden_states = attention_mask.view(-1) == 1
        active_student_hidden_states = [s_state.view(-1, self.student.config.hidden_size)[active_hidden_states] for
                                        s_state in student_hidden_states]
        active_teacher_hidden_states = [t_state.view(-1, self.teacher.config.hidden_size)[active_hidden_states] for
                                        t_state in teacher_hidden_states]

        # attention scores
        student_attention_scores = student_outputs[4]  # List[ B * H * L * L ]
        teacher_attention_scores = teacher_outputs[4]  # List[ B * H * L * L ]
        teacher_attention_scores = [teacher_attention_scores[idx] for idx in self.distill_layer_idx]  # List[ B * H * L * L ]
        bs, seq_len = attention_mask.shape
        attention_mask = torch.matmul(attention_mask.float().unsqueeze(-1),
                                      attention_mask.float().unsqueeze(-2))  # B * L * 1 vs  B * 1 * L = B * L * L
        head_mask = torch.tensor(list(range(bs)) * self.student.config.num_attention_heads)  # [0, 1, ..., B-1] * H(12)
        attention_mask = attention_mask[head_mask]  # (B * H) * L * L
        active_student_attention_scores = [s_score.view(-1, seq_len, seq_len) * attention_mask for s_score in
                                           student_attention_scores]  # (B * H) * L * L
        active_teacher_attention_scores = [t_score.view(-1, seq_len, seq_len) * attention_mask for t_score in
                                           teacher_attention_scores]  # (B * H) * L * L

        # logits
        student_logits = student_outputs[1]  # B*L*V
        teacher_logits = teacher_outputs[1]  # B*L*V
        active_logits = masked_lm_labels.view(-1) != -1
        active_student_logits = student_logits.view(-1, self.tokenizer.vocab_size)[active_logits]
        active_teacher_logits = teacher_logits.view(-1, self.tokenizer.vocab_size)[active_logits]

        loss_mlm = student_outputs[0]
        loss_embd = self.loss_embd_fn(active_student_embedding, active_teacher_embedding)
        loss_hidn_list = [self.loss_hidn_fn(active_student_hidden_states[i], active_teacher_hidden_states[i]) for i in range(len(self.distill_layer_idx))]
        loss_attn_list = [
            self.loss_mse_fn(active_student_attention_scores[i].float(), active_teacher_attention_scores[i].float()) / \
            self.student.config.num_attention_heads for i in range(len(self.distill_layer_idx))]
        loss_pred = (- F.softmax(active_teacher_logits, dim=-1) * F.log_softmax(active_student_logits / self.params.temperature, dim=-1)).sum(-1).mean()

        loss = self.params.alpha_embd * loss_embd + \
               self.params.alpha_hidn * sum(loss_hidn_list) + \
               self.params.alpha_attn * sum(loss_attn_list) + \
               self.params.alpha_pred * loss_pred + \
               self.params.alpha_mlm * loss_mlm

        return loss

    def eval_checkpoint(self):
        """
        evaluate model when
            1. n_step_global % checkpoint_interval == 0
            2. each epoch ends
        """
        if self.is_master and self.do_eval:
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

