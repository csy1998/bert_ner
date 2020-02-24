#!/usr/bin/env bash

# install shannon_preprocessor
#pip install --no-cache-dir -i https://pypi.shannonai.com/root/stable/+simple/ --trusted-host pypi.shannonai.com shannon_preprocessor \

#export PYTHONPATH="$PWD"

# 用来自定义dataset-reader的目录
USER_DIR="/home/mengyuxian/bert_ner/bert_ner/dataset_readers"

#DATA_BIN="/data/nfsdata2/nlp_application/datasets/grammar-correction/chinese/chinese_ner/v2_20191119/bin"
#TASK="ner_v2"

DATA_BIN="/data/nfsdata2/nlp_application/datasets/sentence_compression/BMES_sentence_compression_data/bin"
TASK="detect"

mkdir -p ${DATA_BIN}

for phase in "dev" "train" "test";
    do INFILE=/data/nfsdata2/nlp_application/datasets/sentence_compression/BMES_sentence_compression_data/${phase}.txt.reformat;
    OFILE_PREFIX=${phase};
     shannon-preprocess \
    --output-file $OFILE_PREFIX \
    --input-file $INFILE \
    --destdir $DATA_BIN \
    --workers 16 \
    --user-dir ${USER_DIR} \
    --reader-type "sequence_labeling" \
    --examples ${TASK};
done;