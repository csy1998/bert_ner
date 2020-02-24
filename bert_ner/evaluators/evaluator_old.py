"""
@author: Shuyin Chen
@contact: shuyin_chen@shannonai.com

@version: 1.0
@file: set seed
@time: 2019/11/30 16:50
"""

import os
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler, BatchSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel

from bert_ner.utils.logger import logger
from bert_ner.dataset_readers.load_data import SequenceLabelingDataset
from bert_ner.dataset_readers.ner_reader import TASK2LABELS
from bert_ner.utils.grouped_batch_sampler import GroupedBatchSampler, create_lengths_groups
from bert_ner.metrics.span_f1 import mask_span_f1
from bert_ner.metrics.sbnet_span_f1 import sbnet_span_f1
from bert_ner.losses.loss import ShannonLoss


class Evaluator:
    """
    Evaluator
    """
    def __init__(self,
                 params: dict,
                 model: nn.Module):
        logger.info('Initializing Evaluator')
        self.params = params
        self.data_dir = params.data_dir
        self.multi_gpu = params.multi_gpu
        self.n_gpu = params.n_gpu
        self.is_master = params.is_master
        self.label_list = TASK2LABELS[params.task_name]

        self.model = model

        self.loss = ShannonLoss(self.params.loss_type)

        logger.info(f'Loading data from {self.data_dir}')
        eval_dataset = self.load_mmap_dataset(evaluate=True)

        if params.n_gpu <= 1:
            sampler = SequentialSampler(eval_dataset)
        else:
            sampler = DistributedSampler(eval_dataset)

        if params.group_by_size:
            groups = create_lengths_groups(lengths=eval_dataset.lengths, k=512)
            sampler = GroupedBatchSampler(sampler=sampler, group_ids=groups, batch_size=params.eval_batch_size)
        else:
            sampler = BatchSampler(sampler=sampler, batch_size=params.eval_batch_size, drop_last=False)

        self.eval_dataloader = DataLoader(dataset=eval_dataset,
                                          batch_sampler=sampler,
                                          num_workers=self.params.num_workers,)
                                          #collate_fn=eval_dataset.batch_sequences,)

        self.n_step = 0
        self.total_loss = 0
        self.t_total = len(self.eval_dataloader)
        self.is_master = params.is_master

        if self.multi_gpu:
            logger.info("Using nn.parallel.DistributedDataParallel for distributed training.")
            self.model = DistributedDataParallel(self.model,
                                                 device_ids=[params.local_rank],
                                                 output_device=params.local_rank,
                                                 find_unused_parameters=True)


    def load_mmap_dataset(self, evaluate=False):
        """
        mmap dataset to speed up
        """
        phase = "dev" if evaluate else "train"
        return SequenceLabelingDataset(directory=os.path.join(self.data_dir, "bin"), prefix=phase)


    def eval(self):
        """
        eval model
        return: average loss
        """
        if not self.is_master:
            return

        # if self.is_master:
        logger.info("***** Running evaluating *****")
        logger.info("  Num examples = %d", self.t_total)
        logger.info("  Num Epochs = 1")
        logger.info("  Instantaneous batch size per GPU = %d", self.params.eval_batch_size)
        logger.info("  Total eval batch size (w. parallel, distributed) = %d",
                    self.params.eval_batch_size * self.params.n_gpu)

        self.model.eval()
        preds = None
        masks = None
        out_label_ids = None
        # if self.multi_gpu:
        #     torch.distributed.barrier()

        iter_bar = tqdm(self.eval_dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
        for batch in iter_bar:
            if self.params.n_gpu > 0:
                batch = tuple(t.to(f'cuda:{self.params.local_rank}') for t in batch)
            inputs = {'input_ids': batch[0],
                      'labels': batch[1],
                      'label_mask': batch[2],
                      'attention_mask': batch[3],
                      'token_type_ids': batch[4],
                      }
            outputs = self.model(**inputs)
            logits = outputs[0]
            labels = inputs['labels']
            mask = inputs['label_mask']
            if mask is not None:
                loss_mask = (mask == 1).float()
            else:
                loss_mask = None
            loss = self.loss(logits, labels, loss_mask)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
                masks = inputs['label_mask'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                masks = np.append(masks, inputs['label_mask'].detach().cpu().numpy(), axis=0)


            if self.multi_gpu:
                loss = loss.mean()
            assert loss.item() >= 0
            self.total_loss += loss.item()
            self.last_loss = loss.item()
            self.n_step += 1

            iter_bar.update()
            iter_bar.set_postfix({'loss_cur': f'{self.last_loss:.3f}',
                                  'loss': f'{self.total_loss / self.n_step:.3f}'})

        preds = np.argmax(preds, axis=-1)

        if 'sb_ner' in self.params.task_name:
            results = sbnet_span_f1(preds, out_label_ids, masks, self.label_list)
        else:
            results = mask_span_f1(preds, out_label_ids, masks, self.label_list)

        iter_bar.close()

        return self.total_loss / self.n_step, results


