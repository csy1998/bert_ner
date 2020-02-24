

PYTHONPATH="/home/shuyin/bert_ner":$PYTHONPATH
PYTHONPATH="/home/chenshuyin/bert_ner":$PYTHONPATH
export PYTHONPATH


TASK="sb_ner"
DATA_DIR="/data/nfsdata2/shuyin/data/nested_ner/ace2004/train_dev_test/processed_with_type"
OUTPUT_DIR="/data/nfsdata2/shuyin/model/bert_finetune_models/sb_ner/ace2004/1e-5"


MODEL_TYPE="bert"
LOSS_TYPE="cross_entropy"
MODEL="/data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-12_H-768_A-12"


export CUDA_VISIBLE_DEVICES=0;
export N_GPU=1;


#python ../run_ner.py \
python -m torch.distributed.launch --nproc_per_node=$N_GPU ../../bert_ner/run/run_sbner.py \
    --task_name ${TASK} \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${MODEL} \
    --n_gpu ${N_GPU} \
    --model_type ${MODEL_TYPE} \
    --loss_type ${LOSS_TYPE} \
    --do_train \
    --do_eval \
    --do_lower_case \
    --overwrite_output_dir \
    --train_batch_size 4 \
    --eval_batch_size 16 \
    --biaffine_size 512 \
    --learning_rate 1e-5 \
    --n_epoch 30 \
    --early_stop_patience 30 \
    --last_k_checkpoints 1 \
    --log_interval -1 \
    --checkpoint_interval -1 \
    --warm_up_prop 0. \
    --num_workers 1 \
    --gradient_accumulation_steps 1 \
#    --group_by_size \
#   --fp16 \
#   --fp16_opt_level O2 \

