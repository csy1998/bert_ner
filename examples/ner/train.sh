

PYTHONPATH="/home/shuyin/bert_ner":$PYTHONPATH
PYTHONPATH="/home/chenshuyin/bert_ner":$PYTHONPATH
export PYTHONPATH

#TASK="ner"
#DATA_DIR="/data/nfsdata2/shuyin/data/chinese_ner/msra"
#OUTPUT_DIR="/data/nfsdata2/shuyin/model/electra_finetune_models/chinese_ner_bert_init"

TASK="ner"
DATA_DIR="/data/nfsdata2/shuyin/data/chinese_ner/msra_small"
OUTPUT_DIR="/data/nfsdata2/shuyin/model/bert_finetune_models/chinese_ner_msra_test"


MODEL_TYPE="bert"
LOSS_TYPE="cross_entropy"
MODEL="/data/nfsdata2/nlp_application/models/bert/chinese_L-12_H-768_A-12"
#MODEL="/data/nfsdata2/shuyin/model/electra/test/epoch_0/discriminator"


export CUDA_VISIBLE_DEVICES=3;
export N_GPU=1;


#python ../run_ner.py \
python -m torch.distributed.launch --nproc_per_node=$N_GPU ../../bert_ner/run/run_ner.py \
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
    --train_batch_size 32 \
    --eval_batch_size 8 \
    --learning_rate 2e-5 \
    --n_epoch 10 \
    --early_stop_patience 2 \
    --last_k_checkpoints 2 \
    --log_interval -1 \
    --checkpoint_interval -1 \
    --warm_up_prop 0. \
    --num_workers 1 \
    --gradient_accumulation_steps 1 \
    --group_by_size \
#   --fp16 \
#   --fp16_opt_level O2 \

