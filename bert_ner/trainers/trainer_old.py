"""
@author: Shuyin Chen
@contact: shuyin_chen@shannonai.com

@version: 1.0
@file: set seed
@time: 2019/11/30 14:50
"""

import os
import math
import time
import shutil
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, BatchSampler, DataLoader, SequentialSampler

from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter

from transformers import get_linear_schedule_with_warmup

from bert_ner.utils.logger import logger
from bert_ner.dataset_readers.load_data import SequenceLabelingDataset
from bert_ner.utils.grouped_batch_sampler import GroupedBatchSampler, create_lengths_groups
from bert_ner.evaluators.evaluator_old import Evaluator
from bert_ner.losses.loss import ShannonLoss


class Trainer:
    """
    Trainer
    """
    def __init__(self,
                 params: dict,
                 model: nn.Module,
                 tokenizer: nn.Module):
        logger.info('Initializing Trainer')
        self.params = params
        self.data_dir = params.data_dir
        self.output_dir = params.output_dir
        self.multi_gpu = params.multi_gpu
        self.n_gpu = params.n_gpu
        self.fp16 = params.fp16
        self.do_eval = params.do_eval
        self.is_master = params.is_master
        self.early_stop_patience = params.early_stop_patience

        self.model = model
        self.tokenizer = tokenizer          # only for saving model
        self.model_config = model.config

        self.loss = ShannonLoss(self.params.loss_type)


        logger.info(f'Loading data from {self.data_dir}')
        train_dataset = self.load_mmap_dataset(evaluate=False)
        if self.do_eval:
            self.evaluator = Evaluator(params, model)
            self.eval_loss_epoch = 0
            self.eval_loss_globel = 1e8
            self.eval_span_f1 = 0.
            self.cur_eval_span_f1 = 0.
            self.patience = 0
            self.best_k = -1
            self.save_best_checkpoints = True
            self.last_k = self.params.last_k_checkpoints if self.params.last_k_checkpoints > 0 else self.params.n_epoch
            self.eval_result_path = os.path.join(self.output_dir, 'eval_result.txt')
            with open(self.eval_result_path, 'w') as eval_result_f:
                eval_result_f.write("eval_result: " + '\n')


        if params.n_gpu <= 1:
            # sampler = RandomSampler(train_dataset)
            sampler = SequentialSampler(train_dataset)
        else:
            sampler = DistributedSampler(train_dataset)

        if params.group_by_size:
            groups = create_lengths_groups(lengths=train_dataset.lengths, k=512)
            sampler = GroupedBatchSampler(sampler=sampler, group_ids=groups, batch_size=params.train_batch_size)
        else:
            sampler = BatchSampler(sampler=sampler, batch_size=params.train_batch_size, drop_last=False)

        self.train_dataloader = DataLoader(dataset=train_dataset,
                                           batch_sampler=sampler,
                                           num_workers=self.params.num_workers)

        self.epoch = 0
        self.last_loss = 0
        self.total_loss_epoch = 0
        self.total_loss_globel = 0
        self.n_step_epoch = 0
        self.n_step_globel = 0
        self.n_sequences_epoch = 0
        self.last_log = 0


        logger.info('--- Initializing model optimizer')
        assert params.gradient_accumulation_steps >= 1
        self.num_steps_epoch = len(self.train_dataloader)
        self.t_total = int(
            self.num_steps_epoch / params.gradient_accumulation_steps * params.n_epoch) + 1

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': params.weight_decay},
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
        ]

        if self.is_master:
            logger.info("------ Number of trainable parameters (model): %i" % sum([p.numel() for p in self.model.parameters() if p.requires_grad]))
            logger.info("------ Number of parameters (model): %i" % sum([p.numel() for p in self.model.parameters()]))

        self.optimizer = AdamW(optimizer_grouped_parameters,
                               lr=params.learning_rate,
                               eps=params.adam_epsilon,
                               betas=(0.9, 0.98))

        self.warmup_steps = math.ceil(self.t_total * params.warmup_prop)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=self.warmup_steps,
                                                         num_training_steps=self.t_total)

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            logger.info(f"Using fp16 training: {self.params.fp16_opt_level} level")
            self.model, self.optimizer = amp.initialize(self.model,
                                                        self.optimizer,
                                                        opt_level=self.params.fp16_opt_level)

        if self.multi_gpu:
            logger.info("Using nn.parallel.DistributedDataParallel for distributed training.")
            self.model = DistributedDataParallel(self.model,
                                                 device_ids=[params.local_rank],
                                                 output_device=params.local_rank,
                                                 find_unused_parameters=True)

        if self.is_master:
            logger.info('--- Initializing Tensorboard')
            self.tensorboard = SummaryWriter(log_dir=os.path.join(self.output_dir, 'log', 'train'))
            self.tensorboard.add_text(tag='config/training', text_string=str(self.params), global_step=0)
            self.tensorboard.add_text(tag='config/model', text_string=str(self.model_config), global_step=0)


    def load_mmap_dataset(self, evaluate=False):
        """load_mmap_dataset"""
        phase = "dev" if evaluate else "train"
        return SequenceLabelingDataset(directory=os.path.join(self.data_dir, "bin"), prefix=phase)


    def train(self):
        """
        The real training loop.
        """
        if self.is_master:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", self.num_steps_epoch)
            logger.info("  Num Epochs = %d", self.params.n_epoch)
            logger.info("  Num GPUs = %d", self.params.n_gpu)
            logger.info("  Total warmup steps = %d", self.warmup_steps)
            logger.info("  Total optimization steps = %d", self.t_total)
            logger.info("  Gradient Accumulation steps = %d", self.params.gradient_accumulation_steps)
            logger.info("  Instantaneous batch size per GPU = %d", self.params.train_batch_size)
            logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                        self.params.train_batch_size * self.params.gradient_accumulation_steps * self.params.n_gpu)

        self.last_log = time.time()
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
                                      'loss_epo': f'{self.total_loss_epoch * self.params.gradient_accumulation_steps / self.n_step_epoch:.3f}',
                                      'loss_glo': f'{self.total_loss_globel * self.params.gradient_accumulation_steps / self.n_step_globel:.3f}'})
            iter_bar.close()

            if self.is_master:
                logger.info(f'--- Ending epoch {self.epoch} / {self.params.n_epoch-1}')
            self.end_epoch()

            if self.early_stop_patience > 0 and self.patience > self.early_stop_patience:
                print("training stopped because of early stopping!!!")
                break


    def step(self, inputs):
        """
        One optimization step: forward of student AND teacher, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).

        Input:
        ------
        input_ids: `torch.tensor(bs, seq_length)` - The token ids.
        labels: `torch.tensor(bs, seq_length)`
        label_mask: `torch.tensor(bs, seq_length)`
        attention_mask: `torch.tensor(bs, seq_length)` - The attention mask for self attention.
        token_type_ids: `torch.tensor(bs, seq_length)`
        """
        outputs = self.model(**inputs)
        logits = outputs[0]
        labels = inputs['labels']
        mask = inputs['label_mask']

        # todo 注意 若是bert_crf_tagger 那么 loss 需要在 model 内部计算
        loss = self.calculate_loss(logits, labels, mask)
        assert loss.item() >= 0

        self.n_sequences_epoch += inputs['input_ids'].size(0)
        self.optimize(loss)
        self.end_step()


    def calculate_loss(self, logits=None, labels=None, mask=None):
        """
        calculate loss using self.loss
        """
        if mask is not None:
            loss_mask = (mask == 1).float()
        else:
            loss_mask = None
        loss = self.loss(logits, labels, loss_mask)
        return loss


    def optimize(self, loss):
        """
        Normalization on the loss (gradient accumulation or distributed training), followed by
        backward pass on the loss, possibly followed by a parameter update (depending on the gradient accumulation).
        Also update the metrics for tensorboard.
        """
        # Check for NaN
        if (loss != loss).data.any():
            logger.error('NaN detected')
            exit()

        if self.multi_gpu:
            loss = loss.mean()
        if self.params.gradient_accumulation_steps > 1:
            loss = loss / self.params.gradient_accumulation_steps

        if self.fp16:
            from apex import amp
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        self.last_loss = loss.item()
        self.total_loss_epoch += loss.item()
        self.total_loss_globel += loss.item()

        if (self.n_step_epoch + 1) % self.params.gradient_accumulation_steps == 0:
            if self.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.params.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            self.n_step_epoch += 1
            self.n_step_globel += 1


    def end_step(self):
        """
        write to tensorboard and save checkpoint.
        """
        if self.params.log_interval > 0 and self.n_step_globel % self.params.log_interval == 0:
            self.log_tensorboard()
            self.last_log = time.time()
        if self.params.checkpoint_interval > 0 and self.n_step_globel % self.params.checkpoint_interval == 0:
            self.save_checkpoint()


    def log_tensorboard(self):
        """
        Log into tensorboard. Only by the master process.
        """
        if not self.is_master:
            return

        for param_name, param in self.model.named_parameters():
            self.tensorboard.add_scalar(tag='parameter_mean/' + param_name, scalar_value=param.data.mean(), global_step=self.n_step_globel)
            self.tensorboard.add_scalar(tag='parameter_std/' + param_name, scalar_value=param.data.std(), global_step=self.n_step_globel)
            if param.grad is None:
                continue
            self.tensorboard.add_scalar(tag="grad_mean/" + param_name, scalar_value=param.grad.data.mean(), global_step=self.n_step_globel)
            self.tensorboard.add_scalar(tag="grad_std/" + param_name, scalar_value=param.grad.data.std(), global_step=self.n_step_globel)

        self.tensorboard.add_scalar(tag="losses/cum_avg_loss_epoch", scalar_value=self.total_loss_epoch / self.n_step_epoch, global_step=self.n_step_globel)
        self.tensorboard.add_scalar(tag="losses/loss", scalar_value=self.last_loss, global_step=self.n_step_globel)
        self.tensorboard.add_scalar(tag="learning_rate/lr", scalar_value=self.scheduler.get_lr()[0], global_step=self.n_step_globel)


    def make_dirs(self, path):
        if not os.path.exists(path):
            os.makedirs(path)


    def save_checkpoint(self, save_for_best=False):
        """
        Save the best and last K checkpoints.
        Only by the master process.
        """
        if not self.is_master:
            return

        logger.info(f"******* Saving checkpoints for epoch {self.epoch} *******")
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        if save_for_best == True:
            best_checkpoint_path = os.path.join(self.output_dir, 'checkpoints_best')
            self.make_dirs(best_checkpoint_path)
            model_to_save.save_pretrained(best_checkpoint_path)
            self.tokenizer.save_pretrained(best_checkpoint_path)
            logger.info("Saving ***best*** checkpoint to %s", best_checkpoint_path)

        cur_checkpoint_path = os.path.join(self.output_dir, f'checkpoint_{self.epoch}')
        self.make_dirs(cur_checkpoint_path)
        model_to_save.save_pretrained(cur_checkpoint_path)
        self.tokenizer.save_pretrained(cur_checkpoint_path)
        logger.info("Saving last checkpoint to %s", cur_checkpoint_path)

        if self.epoch >= self.last_k:
            past_checkpoint_path = os.path.join(self.output_dir, f'checkpoint_{self.epoch-self.last_k}')
            shutil.rmtree(past_checkpoint_path, ignore_errors=True)
            logger.info("!!!Removing!!! model checkpoint %s", past_checkpoint_path)

        if self.do_eval:
            with open(self.eval_result_path, 'a') as eval_result_f:
                result = f"epoch: {self.epoch}\t"
                result += f"loss: {self.eval_loss_epoch}\t"
                result += f"span f1: {self.cur_eval_span_f1}\t"
                result += f"best={self.best_k}\t"
                eval_result_f.write(result + '\n')

                if self.epoch == self.params.n_epoch-1:
                    eval_result_f.write(f"\nbest epoch: {self.best_k}\t best span f1: {self.eval_span_f1}\n")

            label_map_path = os.path.join(self.output_dir, 'label_map.txt')
            with open(label_map_path, 'w') as label_map_f:
                for label in self.evaluator.label_list:
                    label_map_f.write(label + '\n')



    def end_epoch(self):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        Do some tensorboard logging and checkpoint saving.
        """
        if self.is_master:
            logger.info(f'{self.n_sequences_epoch} sequences have been trained during this epoch.')

            if self.do_eval:
                save_for_best = False
                print("********** Eval start **********")
                self.eval_loss_epoch, results = self.evaluator.eval()
                self.cur_eval_span_f1 = results['span-f1']
                print("********** Eval end **********")

                print("cur_eval_span_f1: ", self.cur_eval_span_f1)
                print("self.eval_span_f1: ", self.eval_span_f1)

                if self.cur_eval_span_f1 > self.eval_span_f1:
                    self.patience = 0
                    self.eval_span_f1 = self.cur_eval_span_f1
                    self.best_k = self.epoch
                    save_for_best = True

                self.save_checkpoint(save_for_best=save_for_best)
                self.patience += 1

                # print("eval_loss_epoch: ", self.eval_loss_epoch)
                # print("eval_loss_globel: ", self.eval_loss_globel)

                # if self.eval_loss_epoch < self.eval_loss_globel:
                #     self.eval_loss_globel = self.eval_loss_epoch
                #     self.save_checkpoint()

            self.tensorboard.add_scalar(tag='epoch/loss', scalar_value=self.total_loss_epoch/self.n_step_epoch, global_step=self.epoch)

        self.epoch += 1
        self.n_sequences_epoch = 0
        self.n_step_epoch = 0
        self.total_loss_epoch = 0