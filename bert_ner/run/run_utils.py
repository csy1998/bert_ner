# encoding: utf-8
"""
@author: Shuyin Chen
@contact: shuyin_chen@shannonai.com

@version: 1.0
@file: run utils
@time: 2019/12/25 14:50
"""

import os
import json
import shutil
import argparse
from bert_ner.utils.logger import logger


def sanity_checks(args):
    """
    some file sanity checks
    """
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # train output_dir check
    if args.do_train:
        if os.path.exists(args.output_dir):
            if not args.overwrite_output_dir:
                raise ValueError(
                    f'Serialization dir {args.output_dir} already exists, but you have not precised wheter to overwrite it'
                    'Use `--force` if you want to overwrite it')
            else:
                shutil.rmtree(args.output_dir, ignore_errors=True)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        logger.info(f'Experiment will be dumped and logged in {args.output_dir}')

    # eval output_dir check
    if args.do_eval and not args.do_train:
        if not os.path.exists(args.output_dir):
            raise ValueError(f'Model dir {args.output_dir} does not exists!')
        # using output_dir model for evaluating
        args.model_name_or_path = args.output_dir


def get_parser():
    """
    return basic arg parser
    """
    parser = argparse.ArgumentParser(description="Training")

    # for path
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the examples.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--overwrite_output_dir", action='store_true',
                        help="Overwrite dump_path if it already exists.")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")

    # args for training
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--n_epoch", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--early_stop_patience", default=-1, type=int,
                        help="Early stopping, -1 for not using early stopping.")
    parser.add_argument("--save_best_checkpoints", action='store_true',
                        help="Save best checkpoints.")
    parser.add_argument("--last_k_checkpoints", default=1, type=int,
                        help="Save last k checkpoints.")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size")
    parser.add_argument("--group_by_size", action='store_true',
                        help="If true, group sequences that have similar length into the same batch. Default is false.")
    parser.add_argument('--num_workers', type=int, default=1,
                        help="Number of workers used by data loader")

    # for optimizer
    parser.add_argument('--lamb', action='store_true',
                        help="Whether to use lamb optimizer, default using AdamW optimizer.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--warm_up_prop", default=0.05, type=float,
                        help="Linear warm up proportion.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=5.0, type=float,
                        help="Max gradient norm.")

    # for cuda
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--n_gpu", type=int, default=1,
                        help="Number of GPUs in the node.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Distributed training - Local rank")

    # for interval
    parser.add_argument("--log_interval", type=int, default=500,
                        help="Tensorboard logging interval, value -1 for not using it")
    parser.add_argument("--checkpoint_interval", type=int, default=4000,
                        help="Checkpoint interval, value -1 for not using it")

    # for fp16
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    return parser


def save_params(args):
    """
    save params to json file
    """
    if args.is_master and args.do_train:
        logger.info(f'Param: {args}')
        with open(os.path.join(args.output_dir, 'parameters.json'), 'w') as param_f:
            json.dump(vars(args), param_f, indent=4)
