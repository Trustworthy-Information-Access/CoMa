#!/bin/bash
# NOTE: replace ... with actual paths

OUTPUT_DIR=...

BSr=${1-1024}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export PATH=$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WANDB_DISABLED=false
export WANDB_PROJECT=CoMA
export WANDB_API_KEY=...
export WANDB_RUN_GROUP=qwen2_5vl

if [ -z "$TOTAL_COMPRESS_TOKENS" ]; then
    export TOTAL_COMPRESS_TOKENS=32
fi

if [ -z "$N_NODES" ]; then
    export N_NODES=1
fi
if [ -z "$NODE_RANK" ]; then
    export NODE_RANK=0
fi
if [ -z "$MASTER_ADDR" ]; then
    export MASTER_ADDR=127.0.0.1
fi
if [ -z "$MASTER_PORT" ]; then
    export MASTER_PORT=2208
fi

if [ -z "$DATASET_CONFIG" ]; then
    DATASET_CONFIG=scripts/train_ret.yaml
fi

N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

per_device_train_batch_size=$(( $BSr / $N_GPUS / $N_NODES ))

if [ -z "$EXP_NAME" ]; then
    export EXP_NAME=qwen2_5.3B.CoMA.retrv.lora16.avg.BSr${BSr}:64.lr5e5.C${TOTAL_COMPRESS_TOKENS}.steps1500
fi

export WANDB_NAME=$EXP_NAME
export EXP_DIR=$OUTPUT_DIR/$EXP_NAME
export WANDB_DIR=$EXP_DIR

echo $EXP_DIR

if [ -z "$MODEL_PATH" ]; then
    export MODEL_PATH=$OUTPUT_DIR/qwen2_5.3B.CoMA.qa.BS256.lr5e5.C32.epoch1warm0_05/merged
fi

mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/wandb/*

lora_target_modules='q_proj,k_proj,v_proj,o_proj'
echo $lora_target_modules

cmd="torchrun --nnodes=$N_NODES --nproc_per_node=$N_GPUS --master_addr=$MASTER_ADDR --node_rank=$NODE_RANK --master_port=$MASTER_PORT --max_restarts=0 train.py \
    --seed 42 --task_type retrieval \
    --grad_cache --gc_q_chunk_size 8 --gc_p_chunk_size 8 \
    --pooling avg --normalize True --temperature 0.03 \
    --model_name $MODEL_PATH --model_type qwen2_5_vl_compression \
    --lora --lora_r 16 --lora_target_modules $lora_target_modules \
    --bf16 --gradient_accumulation_steps 1 \
    --dataloader_num_workers 16 --dataloader_pin_memory False --dataset_config $DATASET_CONFIG \
    --run_name $EXP_NAME --output_dir $EXP_DIR \
    --max_len 8192 --resize_min_pixels $((28*28*256)) --resize_max_pixels $((28*28*1024)) --image_decay_factor 1 \
    --per_device_train_batch_size $per_device_train_batch_size --interleave_batch_size 64 \
    --warmup_ratio 0.05 --max_steps 1500 --lr_scheduler_type cosine --learning_rate 5e-5 --weight_decay 0 \
    --save_steps 500 --logging_steps 1 --save_safetensors True --remove_unused_columns False \
    --resume_from auto --report_to wandb 2>&1 | tee $EXP_DIR/train.log"
    
echo $cmd
eval $cmd