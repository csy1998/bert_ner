#!/usr/bin/env bash


PYTHONPATH="/home/shuyin/bert_ner":$PYTHONPATH
PYTHONPATH="/home/chenshuyin/bert_ner":$PYTHONPATH
export PYTHONPATH


TASK="ner"
MODEL_TYPE="bert"
LOSS_TYPE="cross_entropy"
DATA_DIR="/data/nfsdata2/shuyin/data/chinese_ner/msra_small"
#EVAL_MODEL_DIR="/data/nfsdata2/shuyin/model/mobilebert_finetune_models/chinese_ner_test/epoch_1"
EVAL_MODEL_DIR="/data/nfsdata2/shuyin/model/bert_init/bert_L-3"
MODEL="/data/nfsdata2/nlp_application/models/bert/chinese_L-12_H-768_A-12"


export CUDA_VISIBLE_DEVICES=2;

python ../../bert_ner/run_ner.py \
    --task_name ${TASK} \
    --data_dir ${DATA_DIR} \
    --output_dir ${EVAL_MODEL_DIR} \
    --model_name_or_path ${MODEL} \
    --loss_type ${LOSS_TYPE} \
    --n_gpu 1 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --n_epoch 1 \
    --do_eval \
    --do_lower_case \
    --model_type ${MODEL_TYPE} \
    --log_interval -1 \
    --checkpoint_interval -1 \
    --group_by_size \

