# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: bert_tagger
@time: 2019/11/19 16:21

    这一行开始写关于本文件的说明与解释
"""


from torch import nn
from transformers import BertPreTrainedModel, BertModel


class BertForSequenceLabeling(BertPreTrainedModel):
    """
    Bert For Sequence Labeling
    """
    def __init__(self, config):
        super(BertForSequenceLabeling, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        # self.apply(self.init_weights)     # pytorch_transformors
        self.init_weights()                 # transformors

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, label_mask=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # return outputs  # logits, (hidden_states), (attentions)
        return logits, sequence_output

