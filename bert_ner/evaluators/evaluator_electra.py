"""
@author: Shuyin Chen
@contact: shuyin_chen@shannonai.com

@version: 1.0
@file: set seed
@time: 2019/11/30 16:50
"""

import os
import copy
from tqdm import tqdm
import numpy as np
from typing import List

import torch
import torch.nn as nn

from bert_ner.utils.logger import logger
from bert_ner.evaluators.evaluator_base import EvaluatorBase
from bert_ner.trainers.trainer_base import TrainerBase


class EvaluatorForElectra(EvaluatorBase):
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
        super(EvaluatorForElectra, self).__init__(params)

        self.model = model

        params_copy = copy.copy(params)
        params_copy.data_dir = "/data/nfsdata2/shuyin/data/chinese_detect/sighan"
        self.eval_dataloader = TrainerBase.load_dataset(params_copy, prefix='sighan', dataset_type="sequence_labeling")
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
            logits = outputs[1]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
                masks = inputs['label_mask'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                masks = np.append(masks, inputs['label_mask'].detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=-1)
        results = self.detect_f1(preds, out_label_ids, masks)

        iter_bar.close()
        logger.info(f"  Num examples = {len(preds)}")

        return results

    def detect_f1(self, batch_preds, batch_labels, batch_masks=None):
        count_detect_hit = 0
        count_wrong_detection = 0
        count_detect_miss = 0

        if batch_masks is None:
            batch_masks = [None] * len(batch_preds)

        for preds, labels, masks in zip(batch_preds, batch_labels, batch_masks):
            if masks is not None:
                preds = self.trunc_by_mask(preds, masks)
                labels = self.trunc_by_mask(labels, masks)

            for i in range(len(preds)):
                if labels[i] == 0:
                    if preds[i] == labels[i]:
                        count_detect_hit += 1
                    else:
                        count_detect_miss += 1
                elif labels[i] == 1:
                    if preds[i] != labels[i]:
                        count_wrong_detection += 1

        detect_precision = count_detect_hit / (count_detect_hit + count_wrong_detection + 1e-10)
        detect_recall = count_detect_hit / (count_detect_hit + count_detect_miss + 1e-10)
        detect_f1 = 2 * detect_precision * detect_recall / (detect_precision + detect_recall + 1e-8)

        return {
            "detect_precision": detect_precision,
            "detect_recall": detect_recall,
            "detect_f1": detect_f1,
        }

    @staticmethod
    def trunc_by_mask(lst: List, masks: List) -> List:
        """根据mask truncate lst"""
        out = []
        for item, mask in zip(lst, masks):
            if mask:
                out.append(item)
        return out

