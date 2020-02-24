#!/bin/sh

export PYTHONPATH=$PYTHONPATH:/home/shuyin/transformers

python ../../bert_ner/utils/extract_bert_model.py \
    --vocab_transform \
    --model_name /data/nfsdata2/shuyin/model/bert/chinese_L-12_H-768_A-12 \
    --dump_checkpoint /data/nfsdata2/shuyin/model/bert_init/weights_4_L.pth \

