# encoding: utf-8
"""
@author: Shuyin Chen
@contact: shuyin_chen@shannonai.com

@version: 1.0
@file: run for ner
@time: 2019/11/30 14:50
"""

from __future__ import absolute_import, division, print_function

import os
import torch

from bert_ner.utils.logger import logger
from bert_ner.utils.init_gpu import init_gpu_params
from bert_ner.utils.set_seed import set_seed

from transformers import BertConfig, BertTokenizer

from bert_ner.models.bert_tagger import BertForSequenceLabeling
from bert_ner.dataset_readers.ner_reader import TASK2LABELS

from bert_ner.trainers.trainer_sbner import TrainerForSBNER
from bert_ner.evaluators.evaluator_sbner import EvaluatorForSBNER
from bert_ner.run.run_utils import get_parser, sanity_checks, save_params
from bert_ner.models.classifier import MultiNonLinearClassifier, NERLabelClassifier, EntityClassifier


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceLabeling, BertTokenizer)
}

def main():
    """
    main function
    """
    parser = get_parser()
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the examples to train selected in the list")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--loss_type", default="cross_entropy", type=str, required=True)
    parser.add_argument("--biaffine_size", default=512, type=int, required=True)
    args = parser.parse_args()

    # ARGS
    init_gpu_params(args)
    set_seed(args)
    if args.is_master:
        sanity_checks(args)
    save_params(args)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # TOKENIZER
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)

    # MODEL
    label_list = TASK2LABELS[args.task_name]
    num_labels = len(label_list) + 4    # add ["X", "[CLS]", "[SEP]", "[PAD]"] to label_list
    config = config_class.from_pretrained(args.model_name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task=args.task_name)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        config=config)

    # Classifier
    args.type_list = []
    type_list_file = os.path.join(args.data_dir, "type_list.txt")
    with open(type_list_file) as type_list_f:
        for line in type_list_f:
            args.type_list.append(line.strip())
    # classifier = MultiNonLinearClassifier(model.config.hidden_size, len(type_list) + 1)

    num_label = len(args.type_list) + 1
    hidden_size = model.config.hidden_size

    ner_label_classifier = NERLabelClassifier(hidden_size,
                                              num_label=num_label)
    entity_classifier = EntityClassifier(hidden_size,
                                         biaffine_size=args.biaffine_size,
                                         num_label=num_label)

    if args.n_gpu > 0:
        model.to(f'cuda:{args.local_rank}')
        ner_label_classifier.to(f'cuda:{args.local_rank}')
        entity_classifier.to(f'cuda:{args.local_rank}')
    logger.info(f'Model loaded from {args.model_name_or_path}.')

    # TRAIN
    if args.do_train:
        torch.cuda.empty_cache()
        trainer = TrainerForSBNER(params=args,
                                  modules=[model, ner_label_classifier, entity_classifier, tokenizer])
        trainer.train()

    # # EVAL
    # if not args.do_train and args.do_eval:
    #     torch.cuda.empty_cache()
    #     evaluator = EvaluatorForSBNER(params=args,
    #                                   model=[model, ner_label_classifier, entity_classifier])
    #     evaluator.eval()


if __name__ == "__main__":
    main()
