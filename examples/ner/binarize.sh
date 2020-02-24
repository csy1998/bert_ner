#!/usr/bin/env bash

# install shannon_preprocessor
#pip install --no-cache-dir -i https://pypi.shannonai.com/root/stable/+simple/ --trusted-host pypi.shannonai.com shannon_preprocessor \


PYTHONPATH="/home/chenshuyin/bert_ner":$PYTHONPATH
PYTHONPATH="/home/shuyin/bert_ner":$PYTHONPATH
export PYTHONPATH

TASK="ner"
TOKENIZER_TYPE="zh"
#USER_DIR="/home/shuyin/bert_ner/bert_ner/dataset_readers"
USER_DIR="/home/chenshuyin/bert_ner/bert_ner/dataset_readers"


DATA_DIR="/data/nfsdata2/shuyin/data/chinese_ner/msra_small"
#DATA_DIR="/data/nfsdata2/shuyin/data/chinese_detect"


READER_TYPE="sequence_labeling"
DATA_BIN=${DATA_DIR}/bin
mkdir -p ${DATA_BIN}


for phase in "train" "dev";
    do INFILE=${DATA_DIR}/${phase}.tsv;
    OFILE_PREFIX=${phase};
    shannon-preprocess \
        --input-file ${INFILE} \
        --output-file ${OFILE_PREFIX} \
        --destdir ${DATA_BIN} \
        --user-dir ${USER_DIR} \
        --reader-type ${READER_TYPE} \
        --task ${TASK} \
        --n_examples 1 \
        --tokenizer_type ${TOKENIZER_TYPE} \
        --workers 4;
done;
