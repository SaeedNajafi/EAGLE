#!/bin/bash

export MASTER_ADDR="$(hostname --fqdn)"
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
export RDVZ_ID=$RANDOM
echo "RDZV Endpoint $MASTER_ADDR:$MASTER_PORT"

run_name="original_eagle_v2_two_layer_bs_2"

NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES="0,1" \
TOKENIZERS_PARALLELISM=false WANDB_MODE=offline accelerate launch -m \
    --main_process_port $MASTER_PORT \
    --main_process_ip $MASTER_ADDR \
    --rdzv_backend static \
    --mixed_precision=bf16 \
    eagle.train.main \
    --basepath /work/saeed-data/model-weights/Llama-3.1-8B-Instruct \
    --tmpdir /work/saeed-data/datasets/eagle_data/sharegpt_0_67999_mufp16 \
    --cpdir /work/saeed-data/checkpoints/eagle_experiments/share_gpt_${run_name} \
    --configpath /home/saeed/EAGLE/eagle/train/EAGLE-LLaMA3-Instruct-8B \
    --num_hidden_layers 2 \
    --add_next_token_loss no \
    --train_lm_head_em_table no \
    --gradient-accumulation-steps 2 \
    --bs 2 \
    --run_name ${run_name} > logs/${run_name}_logs.txt 2>&1
