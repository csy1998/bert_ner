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

from transformers import BertConfig, BertTokenizer

from bert_ner.models.bert_output_attn_score import BertForMaskedLM
from bert_ner.trainers.trainer_tinybert import TrainerForTinyBert
from bert_ner.run.run_utils import get_parser, sanity_checks, save_params


# (tokenizer, stu_config, stu_model, tea_config, tea_model)
MODEL_CLASSES = {
    'tinybert': (BertTokenizer, BertConfig, BertForMaskedLM, BertConfig, BertForMaskedLM),
}


def main():
    """
    main function
    """
    parser = get_parser()
    parser.add_argument("--student_model_path", default=None, type=str, required=True,
                        help="student model path for distillation")
    parser.add_argument("--teacher_model_path", default=None, type=str, required=True,
                        help="teacher model path for distillation")
    parser.add_argument("--alpha_embd", default=1., type=float,
                        help="weight of embedding layer's loss in TinyBert")
    parser.add_argument("--alpha_hidn", default=1., type=float,
                        help="weight of hidden layer's loss in TinyBert")
    parser.add_argument("--alpha_attn", default=1., type=float,
                        help="weight of attention layer's loss in TinyBert")
    parser.add_argument("--alpha_pred", default=1., type=float,
                        help="weight of prediction layer's loss in TinyBert")
    parser.add_argument("--alpha_mlm", default=1., type=float,
                        help="weight of mask language model's loss in TinyBert")
    parser.add_argument("--temperature", default=2., type=float,
                        help="temperature for TinyBert")
    args = parser.parse_args()

    # ARGS
    init_gpu_params(args)
    set_seed(args)
    if args.is_master:
        sanity_checks(args)
    save_params(args)

    # TOKENIZER
    tokenizer = BertTokenizer.from_pretrained(args.teacher_model_path,
                                              do_lower_case=args.do_lower_case)

    # STUDENT
    stu_config = BertConfig.from_pretrained(args.student_model_path,
                                            output_attentions=True,
                                            output_hidden_states=True)
    stu_model = BertForMaskedLM.from_pretrained(args.student_model_path,
                                                config=stu_config)

    # TEACHER
    tea_config = BertConfig.from_pretrained(args.teacher_model_path,
                                            output_attentions=True,
                                            output_hidden_states=True)
    tea_model = BertForMaskedLM.from_pretrained(args.teacher_model_path,
                                                config=tea_config)

    assert stu_model.config.output_attentions
    assert stu_model.config.output_hidden_states
    assert tea_model.config.output_attentions
    assert tea_model.config.output_hidden_states

    if args.n_gpu > 0:
        stu_model.to(f'cuda:{args.local_rank}')
        tea_model.to(f'cuda:{args.local_rank}')

        print("student_config: ", stu_config)
        print("teacher_config: ", tea_config)

        stu_param_num = sum(p.numel() for p in stu_model.parameters())
        tea_param_num = sum(p.numel() for p in tea_model.parameters())
        print("stu_param_num: ", stu_param_num)
        print("tea_param_num: ", tea_param_num)

    logger.info(f'Student model loaded from {args.student_model_path}.')
    logger.info(f'Teacher model loaded from {args.teacher_model_path}.')

    # TRAIN
    if args.do_train:
        torch.cuda.empty_cache()
        trainer = TrainerForTinyBert(params=args,
                                     modules=[stu_model, tea_model, tokenizer])
        if args.is_master: logger.info("Training is starting!")
        trainer.train()
        if args.is_master: logger.info("Training finished!")


if __name__ == "__main__":
    main()
