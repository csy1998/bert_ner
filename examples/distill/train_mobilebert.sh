

PYTHONPATH="/home/shuyin/bert_ner":$PYTHONPATH
PYTHONPATH="/home/chenshuyin/bert_ner":$PYTHONPATH
export PYTHONPATH


DATA_DIR="/data/nfsdata2/shuyin/data/bert_distillation/small_1K"
#DATA_DIR="/data/nfsdata2/shuyin/data/wiki_zh"
OUTPUT_DIR="/data/nfsdata2/shuyin/model/bert_distill_models/mobilebert/test"


### model path
STUDENT="/data/nfsdata2/shuyin/model/bert_google/chinese_L-12_H-768_A-12"
TEACHER="/data/nfsdata2/shuyin/model/bert_google/chinese_L-12_H-768_A-12"



export CUDA_VISIBLE_DEVICES=3;
export N_GPU=1;

python -m torch.distributed.launch --nproc_per_node=$N_GPU ../../bert_ner/run/run_mobilebert.py \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --student_model_path ${STUDENT}\
    --teacher_model_path ${TEACHER}\
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
    --warm_up_prop 0.01 \
    --num_workers 1 \
    --gradient_accumulation_steps 1 \
    --group_by_size \
    --alpha_mlm 0.5 \
    --alpha_pred 0.5\
    --alpha_hidn 10. \
    --alpha_hidn_mean 0.0 \
    --alpha_hidn_var 0.0 \
    --temperature 1. \
    --intra_size 128 \
    --num_ffn 3 \
    --num_attention_heads 4 \
    --pkt_epochs 1 \
    --pd_epochs 10 \
#    --distill \
#   --fp16 \
#   --fp16_opt_level O2 \

