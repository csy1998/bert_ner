

PYTHONPATH="/home/shuyin/bert_ner":$PYTHONPATH
PYTHONPATH="/home/chenshuyin/bert_ner":$PYTHONPATH
export PYTHONPATH


#DATA_DIR="/data/nfsdata2/shuyin/data/bert_distillation/small_1K"
DATA_DIR="/data/nfsdata2/shuyin/data/wiki_zh"
OUTPUT_DIR="/data/nfsdata2/shuyin/model/electra/test"


### model path
#GENERATOR="/data/nfsdata2/shuyin/model/electra/generator_config.json"
#DISCRIMINATOR="/data/nfsdata2/shuyin/model/bert_google/chinese_L-12_H-768_A-12"

GENERATOR="/data/nfsdata2/shuyin/model/electra/bert_init_with_detect/epoch_6/generator"
DISCRIMINATOR="/data/nfsdata2/shuyin/model/electra/bert_init_with_detect/epoch_6/discriminator"



#export CUDA_VISIBLE_DEVICES=0,1,2,3;
#export N_GPU=4;

export CUDA_VISIBLE_DEVICES=0;
export N_GPU=1;

#python ../run_ner.py \
python -m torch.distributed.launch --nproc_per_node=$N_GPU ../../bert_ner/run/run_electra.py \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --generator_path ${GENERATOR}\
    --discriminator_path ${DISCRIMINATOR}\
    --n_gpu ${N_GPU} \
    --do_train \
    --do_eval \
    --do_lower_case \
    --overwrite_output_dir \
    --train_batch_size 24 \
    --eval_batch_size 16 \
    --learning_rate 2e-4 \
    --n_epoch 30 \
    --early_stop_patience -1 \
    --last_k_checkpoints -1 \
    --log_interval -1 \
    --checkpoint_interval -1 \
    --warm_up_prop 0.01 \
    --num_workers 16 \
    --gradient_accumulation_steps 1 \
    --alpha_disc 50. \
    --fp16 \
    --fp16_opt_level O2 \
#    --group_by_size \

