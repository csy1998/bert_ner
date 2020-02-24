#CUDA_VISIBLE_DEVICES=0

OUTPUT_DIR="/data/yuxian/train_logs/20191121_sentence_compression"
DATA_DIR="/data/nfsdata2/nlp_application/datasets/sentence_compression/BMES_sentence_compression_data"
#export CUDA_VISIBLE_DEVICES=1

mkdir -p ${OUTPUT_DIR}

python train_old.py \
--task_name detect \
--data_dir ${DATA_DIR} \
--output_dir ${OUTPUT_DIR} \
--overwrite_output_dir \
--max_seq_length 128 \
--per_gpu_eval_batch_size 32 \
--per_gpu_train_batch_size 32 \
--learning_rate 2e-5 \
--num_train_epochs 5.0 \
--do_train \
--do_eval \
--evaluate_during_training \
--do_lower_case \
--model_type bert \
--logging_steps 288 \
--save_steps 288 \
--model_name_or_path /data/nfsdata2/nlp_application/models/bert/chinese_L-12_H-768_A-12
