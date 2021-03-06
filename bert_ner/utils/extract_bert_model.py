# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocessing script before training DistilBERT.
Specific to BERT -> DistilBERT.
"""
from transformers import BertForMaskedLM, RobertaForMaskedLM
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extraction some layers of the full BertForMaskedLM or RObertaForMaskedLM for Transfer Learned Distillation")
    parser.add_argument("--model_type", default="bert", choices=["bert"])
    parser.add_argument("--model_name", default='/data/nfsdata2/shuyin/model/bert/chinese_L-12_H-768_A-12', type=str)
    parser.add_argument("--dump_checkpoint", default='/data/nfsdata2/shuyin/model/bert_init/weights_4_L.pth', type=str)
    parser.add_argument("--vocab_transform", action='store_true')
    args = parser.parse_args()

    # args.vocab_transform = True

    if args.model_type == 'bert':
        model = BertForMaskedLM.from_pretrained(args.model_name)
        prefix = 'bert'
    else:
        raise ValueError(f'args.model_type should be "bert".')

    state_dict = model.state_dict()
    compressed_sd = {}

    for w in ['word_embeddings', 'position_embeddings']:
        compressed_sd[f'{prefix}.embeddings.{w}.weight'] = \
            state_dict[f'{prefix}.embeddings.{w}.weight']
    for w in ['weight', 'bias']:
        compressed_sd[f'{prefix}.embeddings.LayerNorm.{w}'] = \
            state_dict[f'{prefix}.embeddings.LayerNorm.{w}']

    std_idx = 0
    # for teacher_idx in [0, 2, 4, 7, 9, 11]:
    for teacher_idx in [2, 5, 8, 11]:
        for w in ['weight', 'bias']:
            compressed_sd[f'{prefix}.encoder.layer.{std_idx}.attention.self.query.{w}'] = \
                state_dict[f'{prefix}.encoder.layer.{teacher_idx}.attention.self.query.{w}']
            compressed_sd[f'{prefix}.encoder.layer.{std_idx}.attention.self.key.{w}'] = \
                state_dict[f'{prefix}.encoder.layer.{teacher_idx}.attention.self.key.{w}']
            compressed_sd[f'{prefix}.encoder.layer.{std_idx}.attention.self.value.{w}'] = \
                state_dict[f'{prefix}.encoder.layer.{teacher_idx}.attention.self.value.{w}']

            compressed_sd[f'{prefix}.encoder.layer.{std_idx}.attention.output.dense.{w}'] = \
                state_dict[f'{prefix}.encoder.layer.{teacher_idx}.attention.output.dense.{w}']
            compressed_sd[f'{prefix}.encoder.layer.{std_idx}.attention.output.LayerNorm.{w}'] = \
                state_dict[f'{prefix}.encoder.layer.{teacher_idx}.attention.output.LayerNorm.{w}']

            compressed_sd[f'{prefix}.encoder.layer.{std_idx}.intermediate.dense.{w}'] = \
                state_dict[f'{prefix}.encoder.layer.{teacher_idx}.intermediate.dense.{w}']
            compressed_sd[f'{prefix}.encoder.layer.{std_idx}.output.dense.{w}'] = \
                state_dict[f'{prefix}.encoder.layer.{teacher_idx}.output.dense.{w}']
            compressed_sd[f'{prefix}.encoder.layer.{std_idx}.output.LayerNorm.{w}'] = \
                state_dict[f'{prefix}.encoder.layer.{teacher_idx}.output.LayerNorm.{w}']
        std_idx += 1

    compressed_sd[f'cls.predictions.decoder.weight'] = state_dict[f'cls.predictions.decoder.weight']
    compressed_sd[f'cls.predictions.bias'] = state_dict[f'cls.predictions.bias']
    if args.vocab_transform:
        for w in ['weight', 'bias']:
            compressed_sd[f'cls.predictions.transform.dense.{w}'] = state_dict[f'cls.predictions.transform.dense.{w}']
            compressed_sd[f'cls.predictions.transform.LayerNorm.{w}'] = state_dict[f'cls.predictions.transform.LayerNorm.{w}']

    print(f'N layers selected for distillation: {std_idx}')
    print(f'Number of params transfered for distillation: {len(compressed_sd.keys())}')

    print(f'Save transfered checkpoint to {args.dump_checkpoint}.')
    torch.save(compressed_sd, args.dump_checkpoint)
