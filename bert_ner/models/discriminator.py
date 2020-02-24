# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fei_wang@shannonai.com
@Time       : 2019/8/12 14:48
@Description: Bert for Sequence Labeling
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import BertPreTrainedModel, BertModel


class BertForSequenceLabeling(BertPreTrainedModel):
    r"""
            **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
                Labels for computing the token classification loss.
                Indices should be in ``[0, ..., config.num_labels - 1]``.

        Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
            **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
                Classification loss.
            **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
                Classification scores (before SoftMax).
            **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
                list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
                of shape ``(batch_size, sequence_length, hidden_size)``:
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            **attentions**: (`optional`, returned when ``config.output_attentions=True``)
                list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

        Examples::

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForSequenceLabeling.from_pretrained('bert-base-uncased')
            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
            labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids, labels=labels)
            loss, scores = outputs[:2]

        """

    def __init__(self, config):
        super(BertForSequenceLabeling, self).__init__(config)
        self.num_labels = config.num_labels
        assert self.num_labels == 2
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.discriminator_classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, labels=None, label_mask=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.discriminator_classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # Only keep active parts of the loss
            # loss_fct = F.binary_cross_entropy_with_logits
            loss_fct = CrossEntropyLoss()
            if label_mask is not None:
                loss_mask = label_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[loss_mask]
                active_labels = labels.view(-1)[loss_mask]

                loss = loss_fct(active_logits, active_labels.long())
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
