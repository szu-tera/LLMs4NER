#!/bin/bash
set -euo pipefail

MODEL_PATH=/path/to/your/model
ADAPTER_PATH=/path/to/your/output # This should point to the OUTPUT_DIR from run_lora.sh
DATASET_DIR=/path/to/your/dataset
OUTPUT_DIR=/path/to/your/inference/output
TEMPLATE=llama3 # Specify the template for the model
# Set the test dataset. This name is defined in `data/dataset_info.json`.
EVAL_DATASET=conll03_bracketed_test_alpaca

python -m llamafactory.cli train \
  --stage sft \
  --do_predict \
  --predict_with_generate \
  --model_name_or_path ${MODEL_PATH} \
  --adapter_name_or_path ${ADAPTER_PATH} \
  --eval_dataset ${EVAL_DATASET} \
  --dataset_dir ${DATASET_DIR} \
  --template ${TEMPLATE} \
  --cutoff_len 2048 \
  --overwrite_cache \
  --preprocessing_num_workers 16 \
  --per_device_eval_batch_size 1 \
  --bf16 \
  --output_dir ${OUTPUT_DIR} \
  --overwrite_output_dir \
  --report_to none