"""
@author: Shuyin Chen
@contact: shuyin_chen@shannonai.com

@version: 1.0
@file: set seed
@time: 2019/11/30 14:50
"""

import os
import math
import shutil
import numpy as np
from tqdm import tqdm
from typing import List

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
from bert_ner.dataset_readers.load_data import SequenceLabelingDataset, SequenceMaskingDataset
from bert_ner.utils.grouped_batch_sampler import GroupedBatchSampler, create_lengths_groups

DATASET_CLASSES = {
    'sequence_labeling': SequenceLabelingDataset,
    'sequence_masking': SequenceMaskingDataset,
}


class TrainerBase:
    """
    TrainerBase
    """
    def __init__(self,
                 params: dict):
        """
        提前声明所有变量
        """
        logger.info('Initializing Trainer')
        # params
        self.params = params
        self.data_dir = params.data_dir
        self.output_dir = params.output_dir
        self.multi_gpu = params.multi_gpu
        self.n_gpu = params.n_gpu
        self.fp16 = params.fp16
        self.do_eval = params.do_eval
        self.is_master = params.is_master
        self.early_stop_patience = params.early_stop_patience

        # data loader
        self.train_dataloader = None

        # optimizer
        self.param_optimizer = []
        self.num_steps_epoch = 0
        self.num_steps_total = 0
        self.warm_up_steps = 0
        self.optimizer = None
        self.scheduler = None

        # train
        self.epoch = 0
        self.last_loss = 0
        self.n_step_global = 0
        self.total_loss_global = 0

        # eval
        if self.do_eval:
            self.evaluator = None
            self.eval_results = None
            self.eval_result_cur = 0
            self.eval_result_best = 0
            self.patience = 0
            self.best_k = -1        # best checkpoint number
            self.save_best_checkpoints = True
            self.last_k = self.params.last_k_checkpoints if self.params.last_k_checkpoints > 0 else self.params.n_epoch

            # 每次训练前新建 eval_result.txt
            self.eval_result_path = os.path.join(self.output_dir, 'eval_result.txt')
            with open(self.eval_result_path, 'w') as eval_result_f:
                eval_result_f.write("eval_result: " + '\n')

        # tensor board
        if self.is_master:
            logger.info('--- Initializing Tensorboard')
            self.tensorboard = SummaryWriter(log_dir=os.path.join(self.output_dir, 'log', 'train'))
            self.tensorboard.add_text(tag='config/training', text_string=str(self.params), global_step=0)

    @staticmethod
    def load_dataset(params, prefix='train', dataset_type="sequence_labeling"):
        """
        load_mmap_dataset
        """
        logger.info(f'Loading data from {params.data_dir}')
        dataset_class = DATASET_CLASSES[dataset_type]
        dataset = dataset_class(directory=os.path.join(params.data_dir, "bin"), prefix=prefix)

        if params.n_gpu <= 1:
            # sampler = RandomSampler(dataset)
            sampler = SequentialSampler(dataset)
        else:
            sampler = DistributedSampler(dataset, shuffle=False)

        batch_size = params.train_batch_size if prefix == 'train' else params.eval_batch_size
        if params.group_by_size:
            groups = create_lengths_groups(lengths=dataset.lengths, k=512)
            sampler = GroupedBatchSampler(sampler=sampler, group_ids=groups, batch_size=batch_size)
        else:
            sampler = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)

        dataloader = DataLoader(dataset=dataset,
                                batch_sampler=sampler,
                                num_workers=params.num_workers)
        return dataloader

    def init_optimizer(self, models: List[nn.Module]):
        """
        init optimizer and scheduler
        input:
            models which needs to be optimized
        """
        logger.info('--- Initializing model optimizer')
        assert self.params.gradient_accumulation_steps >= 1
        self.num_steps_epoch = len(self.train_dataloader)
        self.num_steps_total = int(
            self.num_steps_epoch / self.params.gradient_accumulation_steps * self.params.n_epoch) + 1

        for model in models:
            # param_optimizer += list(model.named_parameters())
            self.param_optimizer += list(filter(lambda p: p[-1].requires_grad, model.named_parameters()))

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.param_optimizer if
                        not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': self.params.weight_decay},
            {'params': [p for n, p in self.param_optimizer if
                        any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
        ]

        if self.is_master:
            logger.info("------ Number of trainable parameters (model): %i" % sum(
                [p.numel() for n, p in self.param_optimizer]))
            logger.info("------ Number of parameters (model): %i" % sum(
                [sum([p.numel() for p in m.parameters()]) for m in models]))

        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.params.learning_rate,
                          eps=self.params.adam_epsilon,
                          betas=(0.9, 0.98))

        if self.params.lamb:
            try:
                import apex
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            print("****Using apex.optimizers.FusedLAMB****")
            optimizer = apex.optimizers.FusedLAMB(optimizer_grouped_parameters,
                                                  lr=self.params.learning_rate,
                                                  eps=self.params.adam_epsilon,
                                                  betas=(0.9, 0.999))

        self.warm_up_steps = math.ceil(self.num_steps_total * self.params.warm_up_prop)
        self.scheduler = get_linear_schedule_with_warmup(optimizer,
                                                         num_warmup_steps=self.warm_up_steps,
                                                         num_training_steps=self.num_steps_total)
        return optimizer

    def init_fp16(self, modules: List[nn.Module]):
        """
        init model into fp16
        """
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        logger.info(f"Using fp16 training: {self.params.fp16_opt_level} level")

        fp16_modules = None
        if len(modules) == 1:
            fp16_modules = amp.initialize(modules[0], opt_level=self.params.fp16_opt_level)
        elif len(modules) == 2:
            fp16_modules = amp.initialize(modules[0], modules[1], opt_level=self.params.fp16_opt_level)
        return fp16_modules

    @staticmethod
    def init_ddp(params, model: nn.Module):
        """
        init model into DDP
        """
        logger.info("Using nn.parallel.DistributedDataParallel for distributed training.")
        model = DistributedDataParallel(model,
                                        device_ids=[params.local_rank],
                                        output_device=params.params.local_rank,
                                        find_unused_parameters=True)
        return model

    def train(self):
        """
        train base: log train info
        """
        if self.is_master:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", self.num_steps_epoch)
            logger.info("  Num Epochs = %d", self.params.n_epoch)
            logger.info("  Num GPUs = %d", self.params.n_gpu)
            logger.info("  Total warm up steps = %d", self.warm_up_steps)
            logger.info("  Total optimization steps = %d", self.num_steps_total)
            logger.info("  Gradient Accumulation steps = %d", self.params.gradient_accumulation_steps)
            logger.info("  Instantaneous batch size per GPU = %d", self.params.train_batch_size)
            logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                        self.params.train_batch_size * self.params.gradient_accumulation_steps * self.params.n_gpu)

    def step(self):
        """
        step
        """
        raise NotImplementedError

    def loss(self):
        """
        calculate loss
        """
        raise NotImplementedError

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
        self.total_loss_global += loss.item()

        self.n_step_global += 1
        if self.n_step_global % self.params.gradient_accumulation_steps == 0:
            if self.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.params.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_([p for n, p in self.param_optimizer], self.params.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            # self.n_step_global += 1

    @staticmethod
    def make_dirs(path):
        """
        make_dirs
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def end_step(self):
        """
        write to tensorboard and save checkpoint.
        """
        if self.params.log_interval > 0 and self.n_step_global % self.params.log_interval == 0:
            self.log_tensorboard()
        if self.params.checkpoint_interval > 0 and self.n_step_global % self.params.checkpoint_interval == 0:
            self.eval_checkpoint()

    def end_epoch(self):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        Do some tensor board logging and checkpoint saving.
        """
        self.eval_checkpoint()
        self.epoch += 1

    def log_tensorboard(self):
        """
        Log into tensor board. Only by the master process.
        """
        if not self.is_master:
            return

        for param_name, param in self.param_optimizer:
            self.tensorboard.add_scalar(tag='parameter_mean/' + param_name, scalar_value=param.data.mean(), global_step=self.n_step_global)
            self.tensorboard.add_scalar(tag='parameter_std/' + param_name, scalar_value=param.data.std(), global_step=self.n_step_global)
            if param.grad is None:
                continue
            self.tensorboard.add_scalar(tag="grad_mean/" + param_name, scalar_value=param.grad.data.mean(), global_step=self.n_step_global)
            self.tensorboard.add_scalar(tag="grad_std/" + param_name, scalar_value=param.grad.data.std(), global_step=self.n_step_global)

        self.tensorboard.add_scalar(tag="losses/loss", scalar_value=self.last_loss, global_step=self.n_step_global)
        self.tensorboard.add_scalar(tag="learning_rate/lr", scalar_value=self.scheduler.get_lr()[0], global_step=self.n_step_global)

    def end_eval_checkpoint(self, mode=1):
        """
        Finally arrived at the end of eval checkpoint
        Do checkpoint saving.
        mode:
            1:  save when eval_result_cur > eval_result_best
            -1: save when eval_result_cur < eval_result_best
        """
        save_for_best = False
        whether_to_save = self.eval_result_cur > self.eval_result_best if mode == 1 \
            else self.eval_result_cur < self.eval_result_best
        if whether_to_save:
            self.patience = 0
            self.best_k = self.epoch
            self.eval_result_best = self.eval_result_cur
            save_for_best = True

        self.save_checkpoint(save_for_best=save_for_best)
        self.patience += 1

    def eval_checkpoint(self):
        """
        evaluate model when n_step_global % checkpoint_interval == 0 or each epoch ends
        """
        raise NotImplementedError

    def save_eval_result(self):
        """
        Save eval results
        """
        raise NotImplementedError

    def save_models(self, path):
        """
        Save models
        """
        raise NotImplementedError

    def save_checkpoint(self, save_for_best=False):
        """
        Save the best and last K checkpoints.
        Only by the master process.
        """
        if not self.is_master:
            return
        logger.info(f"******* Saving checkpoints for epoch {self.epoch} *******")

        cur_checkpoint_path = os.path.join(self.output_dir, f'checkpoint_{self.epoch}')
        best_checkpoint_path = os.path.join(self.output_dir, 'checkpoints_best')

        self.make_dirs(cur_checkpoint_path)
        self.make_dirs(best_checkpoint_path)

        # save models
        self.save_models(cur_checkpoint_path)

        # save current checkpoint as best
        if save_for_best:
            self.save_models(best_checkpoint_path)
            logger.info("Saving ***best*** checkpoint to %s", best_checkpoint_path)

        # delete previous checkpoints
        if self.epoch >= self.last_k:
            past_checkpoint_path = os.path.join(self.output_dir, f'checkpoint_{self.epoch - self.last_k}')
            shutil.rmtree(past_checkpoint_path, ignore_errors=True)
            logger.info("!!!Removing!!! model checkpoint %s", past_checkpoint_path)

        # save eval result
        if self.do_eval:
            self.save_eval_result()


