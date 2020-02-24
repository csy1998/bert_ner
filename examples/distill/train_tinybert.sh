

PYTHONPATH="/home/shuyin/bert_ner":$PYTHONPATH
PYTHONPATH="/home/chenshuyin/bert_ner":$PYTHONPATH
export PYTHONPATH


DATA_DIR="/data/nfsdata2/shuyin/data/bert_distillation/small_1K"
#DATA_DIR="/data/nfsdata2/shuyin/data/wiki_zh"
OUTPUT_DIR="/data/nfsdata2/shuyin/model/bert_distill_models/tinybert/test"


### model path
STUDENT_MODEL="/data/nfsdata2/shuyin/model/bert_init/bert_L-4"
TEACHER_MODEL="/data/nfsdata2/shuyin/model/bert_google/chinese_L-12_H-768_A-12"


export CUDA_VISIBLE_DEVICES=3;
export N_GPU=1;

python -m torch.distributed.launch --nproc_per_node=$N_GPU ../../bert_ner/run/run_tinybert.py \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --student_model_path ${STUDENT_MODEL}\
    --teacher_model_path ${TEACHER_MODEL}\
    --n_gpu ${N_GPU} \
    --do_train \
    --do_eval \
    --do_lower_case \
    --overwrite_output_dir \
    --train_batch_size 32 \
    --eval_batch_size 16 \
    --learning_rate 1e-4 \
    --n_epoch 5 \
    --early_stop_patience -1 \
    --last_k_checkpoints 2 \
    --log_interval -1 \
    --checkpoint_interval -1 \
    --warm_up_prop 0. \
    --num_workers 1 \
    --gradient_accumulation_steps 1 \
    --group_by_size \
    --alpha_embd 1. \
    --alpha_hidn 1. \
    --alpha_attn 1.  \
    --alpha_pred 1. \
    --alpha_mlm 0. \
    --temperature 1. \
#   --fp16 \
#   --fp16_opt_level O2 \

