#!/bin/bash
# NOTE: replace ... with actual paths

OUTPUT_DIR=...

BS=${1-256}
per_device_openqa_train_batch_size=${2-1}

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

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

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
    export MASTER_PORT=2207
fi

N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
echo "N_GPUS: $N_GPUS"

if (( BS % (N_GPUS * per_device_openqa_train_batch_size) != 0 )); then
    echo "BS must be divisible by N_GPUS * per_device_openqa_train_batch_size"
    exit 1
fi

export GRAD_ACC=$(( $BS / $N_NODES / $N_GPUS / $per_device_openqa_train_batch_size ))

echo "GRAD_ACC $GRAD_ACC"

if [ -z $DATASET_CONFIG ]; then
    DATASET_CONFIG=scripts/train_qa_qwen2_5vl_3b.yaml
fi

echo "DATASET_CONFIG: $DATASET_CONFIG"

if [ -z $EXP_NAME ]; then
    export EXP_NAME=qwen2_5.7B.CoMA.qa.BS${BS}.lr5e5.C${TOTAL_COMPRESS_TOKENS}.epoch1warm0_05
fi

export WANDB_NAME=$EXP_NAME
export EXP_DIR=$OUTPUT_DIR/$EXP_NAME
export WANDB_DIR=$EXP_DIR
echo "EXP: $EXP_DIR"

mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/wandb/*

cmd="torchrun --nnodes=$N_NODES --nproc_per_node=$N_GPUS --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --max_restarts=0 train.py \
    --bf16 --lora --lora_r 16 --lora_target_modules o_proj,k_proj,q_proj,v_proj \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct --model_type qwen2_5_vl_compression \
    --seed 42 --task_type openqa --multi_turn true \
    --dataloader_num_workers 16 --dataset_config $DATASET_CONFIG \
    --run_name $EXP_NAME --output_dir $EXP_DIR \
    --max_len 8192 --image_decay_factor 1 --resize_min_pixels $((28*28*256)) --resize_max_pixels $((28*28*1024)) \
    --per_device_train_batch_size $per_device_openqa_train_batch_size --gradient_accumulation_steps $GRAD_ACC \
    --lr_scheduler_type cosine --learning_rate 5e-5 --weight_decay 0 --warmup_ratio 0.05 \
    --num_train_epochs 1 --save_steps 500 --logging_steps 1 --save_safetensors true --remove_unused_columns False \
    --resume_from auto --report_to wandb 2>&1 | tee $EXP_DIR/train.log"

echo $cmd
eval $cmd