

PYTHONPATH="/home/shuyin/bert_ner":$PYTHONPATH
PYTHONPATH="/home/chenshuyin/bert_ner":$PYTHONPATH
export PYTHONPATH


DATA_DIR="/data/nfsdata2/shuyin/data/bert_distillation/small_1K"
#DATA_DIR="/data/nfsdata2/shuyin/data/wiki_zh"
OUTPUT_DIR="/data/nfsdata2/shuyin/model/bert_large_model/test"


### model path
MODEL="/data/nfsdata2/nlp_application/models/bert/bert_xunfei/chinese_roberta_wwm_large_ext_pytorch"


export CUDA_VISIBLE_DEVICES=3;
export N_GPU=1;

python -m torch.distributed.launch --nproc_per_node=$N_GPU ../../bert_ner/run/run_finetunebertlarge.py \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --bert_model_path ${MODEL}\
    --n_gpu ${N_GPU} \
    --do_train \
    --do_eval \
    --do_lower_case \
    --overwrite_output_dir \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --n_epoch 3 \
    --early_stop_patience -1 \
    --last_k_checkpoints 1 \
    --log_interval -1 \
    --checkpoint_interval -1 \
    --warm_up_prop 0.01 \
    --num_workers 1 \
    --gradient_accumulation_steps 2 \
    --group_by_size \
#   --fp16 \
#   --fp16_opt_level O2 \

