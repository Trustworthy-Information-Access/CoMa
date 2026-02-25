#!/bin/bash
# ==============================================================================
# Configuration
# ==============================================================================
OUTPUT_BASEDIR="..."
CUDA_VISIBLE_DEVICES="0,1,2,3"
nproc_per_node=$( echo "$CUDA_VISIBLE_DEVICES" | awk -F "," '{print NF}' )
BATCH_SIZE=16
MODALITIES=("image")
DATA_BASEDIR="data/vlm2vec_eval/MMEB-V2"

# ==> Define models and their base output paths here
# Format: "MODEL_NAME;BASE_OUTPUT_PATH"
declare -a MODEL_SPECS
MODEL_SPECS+=( "$OUTPUT_BASEDIR/qwen2_5.3B.CoMA.retrv.lora16.avg.BSr1024:64.lr5e5.C32.steps1500;qwen2_5_vl_compression;$OUTPUT_BASEDIR/Qwen2_5VL-3B-CoMA-C32-LoRA16-BS1024:64" )

# ==============================================================================
# Main Execution Loop
# ==============================================================================
# Loop through each model specification
for spec in "${MODEL_SPECS[@]}"; do
  # Parse the model name and base output path from the spec string
  IFS=';' read -r MODEL_NAME MODEL_BACKBONE BASE_OUTPUT_PATH <<< "$spec"

  echo "================================================="
  echo "🚀 Processing Model: $MODEL_NAME"
  echo "================================================="

  # Loop through each modality for the current model
  for MODALITY in "${MODALITIES[@]}"; do
    DATA_CONFIG_PATH="scripts/eval_$MODALITY.yaml"
    OUTPUT_PATH="$BASE_OUTPUT_PATH/$MODALITY/"

    echo "-------------------------------------------------"
    echo "  - Modality: $MODALITY"
    echo "  - Output Path: $OUTPUT_PATH"

    # Ensure the output directory exists
    mkdir -p "$OUTPUT_PATH"

    cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
      torchrun \
      --nproc_per_node=$nproc_per_node \
      --master_port=2277 \
      --max_restarts=0 \
      eval.py \
      --pooling eos \
      --normalize true \
      --per_device_eval_batch_size $BATCH_SIZE \
      --model_backbone \"$MODEL_BACKBONE\" \
      --model_name \"$MODEL_NAME\" \
      --dataset_config \"$DATA_CONFIG_PATH\" \
      --encode_output_path \"$OUTPUT_PATH\" \
      --data_basedir \"$DATA_BASEDIR\""

    echo "  - Executing command..."
    eval "$cmd"
    echo "  - Done."
    echo "-------------------------------------------------"
  done
done

echo "✅ All jobs completed."
