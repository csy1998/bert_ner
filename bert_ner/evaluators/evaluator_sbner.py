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
from bert_ner.losses.loss import ShannonLoss
from bert_ner.evaluators.evaluator_base import EvaluatorBase
from bert_ner.metrics.sbnet_span_f1 import get_entity, Tag, sbnet_f1
from bert_ner.trainers.trainer_base import TrainerBase


class EvaluatorForSBNER(EvaluatorBase):
    """
    变量声明在 EvaluatorBase 中
    需要重载/重写的方法:
        init
        eval
    """
    def __init__(self,
                 params: dict,
                 model: [nn.Module]):
        logger.info('Initializing Evaluator')
        super(EvaluatorForSBNER, self).__init__(params)
        self.label_list = TASK2LABELS[params.task_name]
        self.type_list = {t: idx for idx, t in enumerate(self.params.type_list)}
        self.type_list['None'] = len(self.type_list)

        self.golden_tags = []
        self.golden_type_map = {}
        with open(os.path.join(self.data_dir, "dev.type")) as train_f:
            for line in train_f:
                instance_id, begin, end, type = line.strip().split()
                tag = Tag(int(instance_id), int(begin), int(end), self.type_list[type])
                self.golden_tags.append(tag)
                self.golden_type_map[tag.to_str()] = tag.type

        self.model, self.ner_label_classifier, self.entity_classifier = model
        self.loss_fn = ShannonLoss(self.params.loss_type)
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
        # self.ner_label_classifier.to(f'cuda:{self.params.local_rank}')
        # self.entity_classifier.to(f'cuda:{self.params.local_rank}')

        pred_tags = []
        pred_types = []
        golden_types = []
        # if self.multi_gpu:
        #     torch.distributed.barrier()

        iter_bar = tqdm(self.eval_dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
        for batch_step, batch in enumerate(iter_bar):
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

            # loss ner
            _, bert_output = outputs
            logits = self.ner_label_classifier(bert_output)
            labels = inputs['labels']
            mask = inputs['label_mask']
            loss_mask = (mask == 1).float() if mask is not None else None
            loss_ner = self.loss_fn(logits, labels, loss_mask)

            preds = logits.detach().cpu().numpy()
            bert_output = bert_output.detach().cpu().numpy()
            masks = mask.detach().cpu().numpy()
            pred_labels = np.argmax(preds, axis=-1)

            # loss cls
            entity_head, entity_tail = [], []
            label_head, label_tail = [], []
            batch_golden_types = []
            instance_cnt = batch_step * self.params.eval_batch_size
            batch_pred_tags = get_entity(pred_labels, masks, self.label_list, instance_cnt)

            for tag in batch_pred_tags:
                pred_entity_str = tag.to_str()
                golden_type = self.golden_type_map[pred_entity_str] \
                    if pred_entity_str in self.golden_type_map.keys() else self.type_list['None']
                batch_golden_types.append(golden_type)

                instance_id = tag.instance_id - instance_cnt
                entity_head += [bert_output[instance_id][tag.begin]]
                entity_tail += [bert_output[instance_id][tag.end]]

                label_head += [pred_labels[instance_id][tag.begin]]
                label_tail += [pred_labels[instance_id][tag.end]]

            entity_head = torch.tensor(entity_head).cuda()
            entity_tail = torch.tensor(entity_tail).cuda()
            label_head = torch.tensor(label_head).cuda()
            label_tail = torch.tensor(label_tail).cuda()

            if len(batch_pred_tags) != 0:
                batch_preds_type_prob = self.entity_classifier(entity_head, entity_tail, label_head, label_tail)
                loss_cls = self.loss_fn(batch_preds_type_prob, torch.tensor(batch_golden_types).cuda())
                assert loss_cls.item() >= 0

                batch_preds_types = np.argmax(batch_preds_type_prob.detach().cpu().numpy(), axis=-1)
                for idx, tag in enumerate(batch_pred_tags):
                    tag.type = batch_preds_types[idx]
                pred_tags += batch_pred_tags
                pred_types += list(batch_preds_types)
                golden_types += batch_golden_types

                loss = loss_ner + loss_cls
            else:
                print("No pred tags!!!")
                loss = loss_ner

            # loss
            assert loss.item() >= 0
            if self.multi_gpu:
                loss = loss.mean()
            self.total_loss += loss.item()
            self.last_loss = loss.item()
            self.n_step += 1

            iter_bar.update()
            iter_bar.set_postfix({'loss_cur': f'{self.last_loss:.3f}',
                                  'loss': f'{self.total_loss / self.n_step:.3f}'})

        # ner / cls / total F1 score
        results = sbnet_f1(pred_tags, self.golden_tags, pred_types, golden_types, self.type_list)
        iter_bar.close()

        return self.total_loss / self.n_step, results

