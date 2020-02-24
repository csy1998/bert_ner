#!/usr/bin/env bash

# install shannon_preprocessor
#pip install --no-cache-dir -i https://pypi.shannonai.com/root/stable/+simple/ --trusted-host pypi.shannonai.com shannon_preprocessor \


PYTHONPATH="/home/shuyin/bert_ner":$PYTHONPATH
PYTHONPATH="/home/chenshuyin/bert_ner":$PYTHONPATH
export PYTHONPATH


#USER_DIR="/home/shuyin/bert_ner/bert_ner/dataset_readers"
USER_DIR="/home/chenshuyin/bert_ner/bert_ner/dataset_readers"

# test data
DATA_DIR="/data/nfsdata2/shuyin/data/bert_distillation/small_1K"

# large data
DATA_DIR="/data/nfsdata2/shuyin/data/wiki_zh"


TOKENIZER_TYPE="zh"
READER_TYPE="sequence_masking"
DATA_BIN=${DATA_DIR}/bin
mkdir -p ${DATA_BIN}


for phase in "train" "dev";
    #do INFILE=${DATA_DIR}/$wiki.{phase}; # for wiki large data
    do INFILE=${DATA_DIR}/${phase}.txt;  # for test data
    OFILE_PREFIX=${phase};
    shannon-preprocess \
        --input-file ${INFILE} \
        --output-file ${OFILE_PREFIX} \
        --destdir ${DATA_BIN} \
        --user-dir ${USER_DIR} \
        --reader-type ${READER_TYPE} \
        --tokenizer_type ${TOKENIZER_TYPE} \
        --workers 4;
done;


