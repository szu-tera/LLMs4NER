#!/bin/bash

set -x

MODEL_PATH=/path/to/your/model
DATASET_DIR=/path/to/your/dataset
OUTPUT_DIR=/path/to/your/output
TEMPLATE=llama3 # Specify the template for the model
# Set the train and dev datasets. These names are defined in `data/dataset_info.json`.
TRAIN_DATASET=conll03_bracketed_train_alpaca
EVAL_DATASET=conll03_bracketed_dev_alpaca

llamafactory-cli train \
  --model_name_or_path ${MODEL_PATH} \
  --trust_remote_code \
  --stage sft \
  --do_train \
  --finetuning_type lora \
  --lora_rank 256 \
  --lora_alpha 512 \
  --lora_target all \
  --dataset ${TRAIN_DATASET} \
  --eval_dataset ${EVAL_DATASET} \
  --dataset_dir ${DATASET_DIR} \
  --template ${TEMPLATE} \
  --cutoff_len 2048 \
  --overwrite_cache \
  --preprocessing_num_workers 16 \
  --dataloader_num_workers 4 \
  --output_dir ${OUTPUT_DIR} \
  --logging_steps 50 \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 3 \
  --eval_strategy steps \
  --eval_steps 500 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 2.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 \
  --report_to none \
  --plot_loss \
  --ddp_timeout 180000000