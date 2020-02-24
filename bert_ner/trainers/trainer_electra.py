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
from bert_ner.evaluators.evaluator_electra import EvaluatorForElectra
from bert_ner.losses.loss import ShannonLoss


class TrainerForElectra(TrainerBase):
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
        super(TrainerForElectra, self).__init__(params)

        self.train_dataloader = self.load_dataset(params, prefix='train', dataset_type="sequence_masking")
        self.generator, self.discriminator, self.tokenizer = modules
        # joint training
        # self.optimizer = self.init_optimizer([self.generator, self.discriminator])
        # froze generator
        self.optimizer = self.init_optimizer([self.discriminator])
        self.softmax_fn = torch.nn.Softmax(dim=-1).cuda()

        if self.fp16:
            self.discriminator, self.optimizer = self.init_fp16([self.discriminator, self.optimizer])
            # self.generator = self.init_fp16([self.generator])
            self.generator = self.generator.half()

        if self.multi_gpu:
            self.discriminator = self.init_ddp(params, self.discriminator)

        if self.do_eval:
            self.evaluator = EvaluatorForElectra(params, self.discriminator)

    def train(self):
        """
        The real training loop.
        """
        # log train info
        super().train()

        # self.generator.train()
        self.generator.eval()
        self.discriminator.train()
        self.generator.zero_grad()
        self.discriminator.zero_grad()

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
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        masked_lm_labels = inputs['masked_lm_labels']

        generator_outputs = self.generator(**inputs)
        loss_mlm = generator_outputs[0]

        generator_scores = generator_outputs[1]  # .clone()                                 # BL * V
        generator_scores = generator_scores.view(-1, generator_scores.shape[-1])  # BL * V
        generator_prob = self.softmax_fn(generator_scores)  # BL * V

        lm_label_mask = masked_lm_labels.view(-1) != -1  # BL
        label_generator_prob = generator_prob[lm_label_mask]  # num_mask * V
        word_sampler = torch.distributions.Categorical(label_generator_prob)
        sample_words = word_sampler.sample()  # num_mask * 1

        disc_input_ids = input_ids.clone().view(-1)
        assert lm_label_mask.sum() == sample_words.shape[0]
        disc_input_ids[lm_label_mask] = sample_words

        detect_labels = (disc_input_ids == input_ids.view(-1)).float()
        assert (detect_labels >= 0.).sum().item() == (detect_labels <= 1.).sum().item()

        detect_label_mask = attention_mask.clone()
        detect_label_mask = F.pad(detect_label_mask, (-1, 1))
        detect_label_mask[:, 0] = 0  # [0,1,1,1,0,...,0,0,0]

        input_size = input_ids.shape
        discriminator_outputs = self.discriminator(input_ids=disc_input_ids.view(input_size),
                                                   attention_mask=attention_mask,
                                                   labels=detect_labels,
                                                   label_mask=detect_label_mask)

        loss_disc = discriminator_outputs[0]
        loss = loss_mlm + self.params.alpha_disc * loss_disc
        print('loss_mlm: ', loss_mlm)
        print('loss_disc: ', loss_disc)
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
            with torch.no_grad:
                self.eval_results = self.evaluator.eval()
            self.eval_result_cur = self.eval_results['detect_f1']

            print("cur_detect_f1: ", self.eval_result_cur)
            print("detect_f1: ", self.eval_result_best)

            self.end_eval_checkpoint()

    def save_models(self, path):
        """
        save models
        """
        model = self.generator
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(path)

        model = self.discriminator
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(path)

        self.tokenizer.save_pretrained(path)

    def save_eval_result(self):
        """
        Save eval results
        """
        with open(self.eval_result_path, 'a') as eval_result_f:
            result = f"epoch: {self.epoch} \t"
            result += f"detect_precision: {self.eval_results['detect_precision']} \t"
            result += f"detect_recall: {self.eval_results['detect_recall']} \t"
            result += f"detect_f1: {self.eval_results['detect_f1']} \t"
            result += f"detect_f1_best={self.eval_result_best} \t"
            result += f"best={self.best_k} \t"
            eval_result_f.write(result + '\n')

            if self.epoch == self.params.n_epoch - 1:
                result = f"best epoch: {self.best_k} \t"
                result += f"detect_f1_best: {self.eval_result_best} \t"
                eval_result_f.write('\n' + result + '\n')

