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
from bert_ner.dataset_readers.ner_reader import TASK2LABELS
from bert_ner.metrics.span_f1 import mask_span_f1
from bert_ner.losses.loss import ShannonLoss
from bert_ner.evaluators.evaluator_base import EvaluatorBase
from bert_ner.trainers.trainer_base import TrainerBase


class EvaluatorForNER(EvaluatorBase):
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
        super(EvaluatorForNER, self).__init__(params)
        self.label_list = TASK2LABELS[params.task_name]

        self.model = model
        self.loss = ShannonLoss(self.params.loss_type)
        self.eval_dataloader = TrainerBase.load_dataset(params, prefix='dev', dataset_type="sequence_labeling")
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
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs[0]
            labels = inputs['labels']
            mask = inputs['label_mask']
            if mask is not None:
                loss_mask = (mask == 1).float()
            else:
                loss_mask = None
            loss = self.loss(logits, labels, loss_mask)

            preds_batch = np.argmax(logits.detach().cpu().numpy(), axis=-1)
            if preds is None:
                preds = preds_batch
                out_label_ids = inputs['labels'].detach().cpu().numpy()
                masks = inputs['label_mask'].detach().cpu().numpy()
            else:
                preds = np.append(preds, preds_batch, axis=0)
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

        # preds = np.argmax(preds, axis=-1)

        results = mask_span_f1(preds, out_label_ids, masks, self.label_list)
        iter_bar.close()

        return self.total_loss / self.n_step, results


