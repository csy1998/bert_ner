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

from bert_ner.trainers.trainer_electra import TrainerForElectra
# from bert_ner.models.bert_tagger import BertForSequenceLabeling
from bert_ner.models.discriminator import BertForSequenceLabeling
from bert_ner.run.run_utils import get_parser, sanity_checks, save_params

# (tokenizer, gen_config, gen_model, disc_config, disc_model)
MODEL_CLASSES = {
    'electra': (BertTokenizer, BertConfig, BertForMaskedLM, BertConfig, BertForSequenceLabeling),
}


def main():
    """
    main function
    """
    parser = get_parser()
    parser.add_argument("--generator_path", default=None, type=str, required=True,
                        help="student model path for distillation")
    parser.add_argument("--discriminator_path", default=None, type=str, required=True,
                        help="teacher model path for distillation")
    parser.add_argument("--alpha_disc", default=50., type=float,
                        help="weight of discriminator's loss in Electra")
    args = parser.parse_args()

    # ARGS
    init_gpu_params(args)
    set_seed(args)
    if args.is_master:
        sanity_checks(args)
    save_params(args)

    # TOKENIZER
    tokenizer = BertTokenizer.from_pretrained(args.discriminator_path,
                                              do_lower_case=args.do_lower_case)

    # GENERATOR
    # generator_config = BertConfig.from_json_file(args.generator_path)
    # generator = BertForMaskedLM(config=generator_config)
    generator = BertForMaskedLM.from_pretrained(args.generator_path)

    # DISCRIMINATOR
    # random init
    # discriminator_config = disc_config_class.from_pretrained(args.discriminator_path)
    # discriminator = disc_model_class(config=discriminator_config)

    # bert init
    discriminator = BertForSequenceLabeling.from_pretrained(args.discriminator_path)

    if args.n_gpu > 0:
        generator.to(f'cuda:{args.local_rank}')
        discriminator.to(f'cuda:{args.local_rank}')

        # print("generator_config: ", generator.config)
        # print("discriminator_config: ", discriminator.config)

        gen_param_num = sum(p.numel() for p in generator.parameters())
        disc_param_num = sum(p.numel() for p in discriminator.parameters())
        print("generator_param_num: ", gen_param_num)
        print("discriminator_param_num: ", disc_param_num)

    logger.info(f'generator config loaded from {args.generator_path}.')
    logger.info(f'discriminator loaded from {args.discriminator_path}.')

    # TRAIN
    if args.do_train:
        torch.cuda.empty_cache()
        trainer = TrainerForElectra(params=args,
                                    modules=[generator, discriminator, tokenizer])
        if args.is_master: logger.info("Training is starting!")
        trainer.train()
        if args.is_master: logger.info("Training finished!")


if __name__ == "__main__":
    main()
