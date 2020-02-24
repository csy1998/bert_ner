# encoding: utf-8
"""
@author: Shuyin Chen
@contact: shuyin_chen@shannonai.com

@version: 1.0
@file: set seed
@time: 2019/11/30 14:50
"""

from __future__ import absolute_import, division, print_function

import os
import torch

from bert_ner.utils.logger import logger
from bert_ner.utils.init_gpu import init_gpu_params
from bert_ner.utils.set_seed import set_seed

from transformers import BertConfig, BertTokenizer, BertForMaskedLM

from bert_ner.trainers.trainer_finetunebertlarge import TrainerForFineTuneBertLarge
from bert_ner.run.run_utils import get_parser, sanity_checks, save_params

# (tokenizer, stu_config, stu_model, tea_config, tea_model)
MODEL_CLASSES = {
    'bert_large': (BertTokenizer, BertConfig, BertForMaskedLM),
}


def main():
    """
    main function
    """
    parser = get_parser()
    parser.add_argument("--bert_model_path", default=None, type=str, required=True,
                        help="teacher model path for distillation")
    args = parser.parse_args()

    # ARGS
    init_gpu_params(args)
    set_seed(args)
    if args.is_master:
        sanity_checks(args)
    save_params(args)

    # TOKENIZER
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path,
                                              do_lower_case=args.do_lower_case)

    # MODEL
    config = BertConfig.from_pretrained(args.bert_model_path,
                                        output_attentions=True,
                                        output_hidden_states=True)
    bert_model = BertForMaskedLM.from_pretrained(args.bert_model_path,
                                                 config=config)
    assert bert_model.config.output_attentions
    assert bert_model.config.output_hidden_states

    if args.n_gpu > 0:
        bert_model.to(f'cuda:{args.local_rank}')
        model_param_num = sum(p.numel() for p in bert_model.parameters())
        print("model config: ", config)
        print("stu_param_num: ", model_param_num)

    for name, param in bert_model.named_parameters():
        if 'predictions' in name:
            param.requires_grad = True
            print(name, param.requires_grad)
        else:
            param.requires_grad = False

    logger.info(f'Bert model loaded from {args.bert_model_path}.')

    # TRAIN
    if args.do_train:
        torch.cuda.empty_cache()
        trainer = TrainerForFineTuneBertLarge(params=args,
                                              modules=[bert_model, tokenizer])
        if args.is_master: logger.info("Training is starting!")
        trainer.train()
        if args.is_master: logger.info("Training finished!")


if __name__ == "__main__":
    main()
