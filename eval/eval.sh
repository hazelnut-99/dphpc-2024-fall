#!/bin/bash

# Init environment
shopt -s expand_aliases
source ~/.bashrc
env_init
rm -rf checkpoints

# Generate the initial weights randomly and train the model for 10 steps
# Vanilla Nanotron training
cd ../src/nanotron-vanilla
pip install -e . > /dev/null 2>&1
cd - > /dev/null 2>&1
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=2 ../src/nanotron-vanilla/run_train.py --config-file config_tiny_llama_eval_vanilla.yaml &> log_vanilla.txt
echo "[INFO] Vanilla Nanotron training done"

# Modified Nanotron training
cd ../src/nanotron-sp
pip install -e . > /dev/null 2>&1
cd - > /dev/null 2>&1
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=2 ../src/nanotron-sp/run_train.py --config-file config_tiny_llama_eval_sp.yaml &> log_sp.txt
echo "[INFO] Modified Nanotron training done"

