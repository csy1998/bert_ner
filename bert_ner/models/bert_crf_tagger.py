# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: bert_crf
@time: 2019/11/11 20:33

    这一行开始写关于本文件的说明与解释
"""


from torch import nn
from transformers import BertPreTrainedModel, BertModel
from bert_ner.models.crf import CRF


class BertCrfForSequenceLabeling(BertPreTrainedModel):
    """Bert For Sequence Labeling"""

    def __init__(self, config):
        super(BertCrfForSequenceLabeling, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels + 2)
        self.crf = CRF(**{"target_size": self.config.num_labels,
                          "use_cuda": True,
                          "average_batch": True})

        # self.apply(self.init_weights)     # pytorch_transformors
        self.init_weights()                 # transformors

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, label_mask=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        # logits = logits[:, 1:, :]  # 不让CLS参与forward

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        # path_score, best_path = self.crf(logits, attention_mask)
        if labels is not None:
            crf_label_mask = input_ids != 0
            loss = self.crf.neg_log_likelihood_loss(logits, crf_label_mask, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
