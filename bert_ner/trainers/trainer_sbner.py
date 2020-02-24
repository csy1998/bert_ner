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
from tqdm import tqdm
import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_ner.utils.logger import logger
from bert_ner.trainers.trainer_base import TrainerBase
from bert_ner.evaluators.evaluator_sbner import EvaluatorForSBNER
from bert_ner.losses.loss import ShannonLoss

# import for sb_ner
from bert_ner.metrics.sbnet_span_f1 import get_entity, Tag


class TrainerForSBNER(TrainerBase):
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
        super(TrainerForSBNER, self).__init__(params)

        self.train_dataloader = self.load_dataset(params, prefix='train', dataset_type="sequence_labeling")
        self.model, self.ner_label_classifier, self.entity_classifier, self.tokenizer = modules
        self.optimizer = self.init_optimizer([self.model])
        self.loss_fn = ShannonLoss(self.params.loss_type)

        self.type_list = {t: idx for idx, t in enumerate( self.params.type_list)}
        self.type_list['None'] = len(self.type_list)
        print("self.type_list: ", self.type_list)

        self.golden_tags = []
        self.golden_type_map = {}
        with open(os.path.join(self.data_dir, "dev.type")) as train_f:
            for line in train_f:
                instance_id, begin, end, type = line.strip().split()
                tag = Tag(instance_id, begin, end, self.type_list[type])
                self.golden_tags.append(tag)
                self.golden_type_map[tag.to_str()] = tag.type

        if self.fp16:
            self.model, self.optimizer = self.init_fp16([self.model, self.optimizer])

        if self.multi_gpu:
            self.model = self.init_ddp(params, self.model)

        if self.do_eval:
            self.eval_loss = 0
            self.evaluator = EvaluatorForSBNER(params, [self.model, self.ner_label_classifier, self.entity_classifier])
            self.eval_result_best = {'ner': {'precision': 0, 'recall': 0, 'f1': 0},
                                     'cls': 0,
                                     'total': {'precision': 0, 'recall': 0, 'f1': 0}}

        self.ner_label_classifier.to(f'cuda:{self.params.local_rank}')
        self.entity_classifier.to(f'cuda:{self.params.local_rank}')

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
            for batch_step, batch in enumerate(iter_bar):
                if self.params.n_gpu > 0:
                    batch = tuple(t.to(f'cuda:{self.params.local_rank}') for t in batch)
                inputs = {'input_ids': batch[0],
                          'labels': batch[1],
                          'label_mask': batch[2],
                          'token_type_ids': batch[4],
                          'attention_mask': batch[3],
                          }
                self.step(inputs, batch_step)

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

    def step(self, inputs=None, batch_step=None):
        """
        One optimization step
        Input:
            input_ids: `torch.tensor(bs, seq_length)` - The token ids.
            labels: `torch.tensor(bs, seq_length)`
            label_mask: `torch.tensor(bs, seq_length)`
            attention_mask: `torch.tensor(bs, seq_length)` - The attention mask for self attention.
        """
        outputs = self.model(**inputs)
        _, bert_output = outputs

        # loss ner
        logits = self.ner_label_classifier(bert_output)
        labels = inputs['labels']
        mask = inputs['label_mask']
        loss_ner = self.loss(logits, labels, mask)
        assert loss_ner.item() >= 0

        # loss cls
        # preds = logits.detach().cpu().numpy()
        # bert_output = bert_output.detach().cpu().numpy()
        # masks = mask.detach().cpu().numpy()
        #
        # entities, entity_head, entity_tail = [], [], []
        # label_head, label_tail = [], []
        # batch_golden_types = []
        # pred_labels = np.argmax(preds, axis=-1)
        # instance_cnt = batch_step * self.params.train_batch_size
        # batch_pred_tags = get_entity(pred_labels, masks, self.evaluator.label_list, instance_cnt)
        #
        # for tag in batch_pred_tags:
        #     pred_entity_str = tag.to_str()
        #     golden_type = self.golden_type_map[pred_entity_str] \
        #         if pred_entity_str in self.golden_type_map.keys() else self.type_list['None']
        #     batch_golden_types.append(golden_type)
        #
        #     instance_id = tag.instance_id - instance_cnt
        #     entity_head += [bert_output[instance_id][tag.begin]]
        #     entity_tail += [bert_output[instance_id][tag.end]]
        #     # entities += [bert_output[instance_id][tag.begin:tag.end].mean(axis=0)]
        #
        #     label_head += [pred_labels[instance_id][tag.begin]]
        #     label_tail += [pred_labels[instance_id][tag.end]]
        #
        # entity_head = torch.tensor(entity_head).cuda()
        # entity_tail = torch.tensor(entity_tail).cuda()
        # label_head = torch.tensor(label_head).cuda()
        # label_tail = torch.tensor(label_tail).cuda()
        #
        # if len(batch_pred_tags) != 0:
        #     batch_preds_type_logits = self.entity_classifier(entity_head, entity_tail, label_head, label_tail)
        #     loss_cls = self.loss_fn(batch_preds_type_logits, torch.tensor(batch_golden_types).cuda())
        #     assert loss_cls.item() >= 0
        #     loss = loss_ner + loss_cls
        # else:
        #     print("No pred tags!!!")
        #     loss = loss_ner

        loss = loss_ner
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
            self.eval_loss, self.eval_result_cur = self.evaluator.eval()

            print('eval_result: ', self.eval_result_cur)
            print("eval_f1_cur: ", self.eval_result_cur['total']['f1'])
            print("eval_f1_best: ", self.eval_result_best['total']['f1'])

            save_for_best = False
            if self.eval_result_cur['total']['f1'] > self.eval_result_best['total']['f1']:
                self.patience = 0
                self.eval_result_best = self.eval_result_cur
                self.best_k = self.epoch
                save_for_best = True

            self.save_checkpoint(save_for_best=save_for_best)
            self.patience += 1

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
            result += f"ner f1: {self.eval_result_cur['ner']['f1']}\t"
            result += f"cls acc: {self.eval_result_cur['cls']}\t"
            result += f"total f1: {self.eval_result_cur['total']['f1']}\t"
            result += f"best={self.best_k} \t"
            eval_result_f.write(result + '\n')

            if self.epoch == self.params.n_epoch - 1:
                result = f"best epoch: {self.best_k} \t"
                result += f"best f1: {self.eval_result_best['total']['f1']} \t"
                eval_result_f.write('\n' + result + '\n')

