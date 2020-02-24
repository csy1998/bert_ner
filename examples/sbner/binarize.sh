#!/usr/bin/env bash

# install shannon_preprocessor
#pip install --no-cache-dir -i https://pypi.shannonai.com/root/stable/+simple/ --trusted-host pypi.shannonai.com shannon_preprocessor \


PYTHONPATH="/home/chenshuyin/bert_ner":$PYTHONPATH
PYTHONPATH="/home/shuyin/bert_ner":$PYTHONPATH
export PYTHONPATH


TASK="sb_ner"
TOKENIZER_TYPE="en"
#USER_DIR="/home/shuyin/bert_ner/bert_ner/dataset_readers"
USER_DIR="/home/chenshuyin/bert_ner/bert_ner/dataset_readers"


DATA_DIR="/data/nfsdata2/shuyin/data/nested_ner/ace2004/train_dev_test/processed_with_type"
#DATA_DIR="/data/nfsdata2/shuyin/data/nested_ner/ace2005/train_dev_test/processed"


READER_TYPE="sequence_labeling"
DATA_BIN=${DATA_DIR}/bin
mkdir -p ${DATA_BIN}


for phase in "train" "dev" "test";
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
