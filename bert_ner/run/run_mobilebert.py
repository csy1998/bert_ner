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

from bert_ner.models.bert_output_attn_score_with_stage import BertForMaskedLM
from bert_ner.models.bert_mobile import BertForMaskedLM as MobileBertForMaskedLM
from bert_ner.models.bert_mobile_config import BertConfig as MobileBertConfig

from bert_ner.trainers.trainer_mobilebert import TrainerForMobileBert
from bert_ner.run.run_utils import get_parser, sanity_checks, save_params


# (tokenizer, stu_config, stu_model, tea_config, tea_model)
MODEL_CLASSES = {
    'mobilebert': (BertTokenizer, MobileBertConfig, MobileBertForMaskedLM, BertConfig, BertForMaskedLM),
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

    parser.add_argument("--alpha_pred", default=1., type=float,
                        help="weight of prediction layer's loss in MobileBert")
    parser.add_argument("--alpha_mlm", default=1., type=float,
                        help="weight of mask language model's loss in MobileBert")
    parser.add_argument("--alpha_hidn", default=1., type=float,
                        help="weight of hidden layer's loss in MobileBert")
    parser.add_argument("--alpha_hidn_mean", default=1., type=float,
                        help="weight of hidden layer's mean loss in MobileBert")
    parser.add_argument("--alpha_hidn_var", default=1., type=float,
                        help="weight of hidden layer's variance loss in MobileBert")
    parser.add_argument("--temperature", default=2., type=float,
                        help="temperature in MobileBert")
    parser.add_argument('--intra_size', default=192, type=int,
                        help="intra size for bottle neck")
    parser.add_argument('--num_ffn', default=4, type=int,
                        help="num of FFN module in each bert layer")
    parser.add_argument('--num_attention_heads', default=4, type=int,
                        help="num of attention_heads in each bert layer")
    parser.add_argument('--pkt_epochs', default=1, type=int,
                        help="num of epochs in progressive knowledge distillation")
    parser.add_argument('--pd_epochs', default=10, type=int,
                        help="num of epochs in prediction distillation")
    parser.add_argument('--distill', action="store_true",
                        help="whether to do distillation in last layer")
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

    # init model
    if not args.distill:
        stu_config = MobileBertConfig.from_json_file(os.path.join(args.student_model_path, 'config.json'))
        # stu_config = MobileBertConfig.from_pretrained(os.path.join(args.student_model_path, 'config.json'))

        # todo(shuyin) 我也不知道为啥, MobileBertConfig.from_pretrained之后 vocab_size 没有保持一致 所以令 vocab_size = 21128
        stu_config.vocab_size = 21128
        stu_config.output_hidden_states = True
        stu_config.intra_size = args.intra_size
        stu_config.intermediate_size = args.intra_size * 4
        stu_config.num_ffn = args.num_ffn
        stu_config.num_attention_heads = args.num_attention_heads
        print("stu_config: ", stu_config)

        stu_model = MobileBertForMaskedLM(config=stu_config)
        tea_config = BertConfig.from_pretrained(args.teacher_model_path,
                                                output_hidden_states=True)
        tea_model = BertForMaskedLM.from_pretrained(args.teacher_model_path,
                                                    config=tea_config)

        embeddings_names = ['bert.embeddings.word_embeddings.weight',
                            'bert.embeddings.position_embeddings.weight',
                            'bert.embeddings.position_embeddings.weight',
                            'bert.embeddings.LayerNorm.weight',
                            'bert.embeddings.LayerNorm.bias', ]

        # init student model's embedding layer with teacher's weights
        embeddings_params = {k: v for k, v in tea_model.state_dict().items() if k in embeddings_names}
        student_model_state = stu_model.state_dict()
        student_model_state.update(embeddings_params)
        stu_model.load_state_dict(student_model_state)

        for w in embeddings_names:
            assert torch.all(torch.eq(tea_model.state_dict()[w], stu_model.state_dict()[w]))

        assert stu_model.config.output_hidden_states == True
        assert tea_model.config.output_hidden_states == True

    # init model finished. do last distillation
    else:
        args.student_model_path = "/data/nfsdata2/shuyin/model/bert_distill_models/mobilebert/test_init/mobilebert_init"
        stu_config = MobileBertConfig.from_pretrained(args.student_model_path)
        stu_config.vocab_size = 21128
        stu_model = MobileBertForMaskedLM.from_pretrained(args.student_model_path,
                                                          config=stu_config)
        # stu_model = MobileBertForMaskedLM.from_pretrained(args.student_model_path)
        tea_model = BertForMaskedLM.from_pretrained(args.teacher_model_path)

    if args.n_gpu > 0:
        stu_model.to(f'cuda:{args.local_rank}')
        tea_model.to(f'cuda:{args.local_rank}')

        print("student_config: ", stu_model.config)
        print("teacher_config: ", tea_model.config)

        stu_param_num = sum(p.numel() for p in stu_model.parameters())
        tea_param_num = sum(p.numel() for p in tea_model.parameters())
        print("stu_param_num: ", stu_param_num)
        print("tea_param_num: ", tea_param_num)

    logger.info(f'Student model loaded from {args.student_model_path}.')
    logger.info(f'Teacher model loaded from {args.teacher_model_path}.')

    # TRAIN
    if args.do_train:
        torch.cuda.empty_cache()
        trainer = TrainerForMobileBert(params=args,
                                       modules=[stu_model, tea_model, tokenizer])
        if args.is_master: logger.info("Training is starting!")
        trainer.train()
        if args.is_master: logger.info("Training finished!")


if __name__ == "__main__":
    main()
