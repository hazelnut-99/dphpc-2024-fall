#!/bin/bash

# Init environment
shopt -s expand_aliases
source ~/.bashrc
env_init
rm -rf checkpoints

# Generate the initial weights randomly and train the model for 10 steps
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=2 ../src/nanotron-vanilla/run_train.py --config-file config_tiny_llama_init.yaml > /dev/null 2>&1
echo "[INFO] Initial weights generated"

# Copy the initial weights to the resume directory for both the vanilla and the modified model
cp -r checkpoints/init checkpoints/vanilla
cp -r checkpoints/init checkpoints/sp

# Vanilla Nanotron training
cd ../src/nanotron-vanilla
pip install -e . > /dev/null 2>&1
cd - > /dev/null 2>&1
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=2 ../src/nanotron-vanilla/run_train.py --config-file config_tiny_llama_resume_vanilla.yaml > /dev/null 2>&1
echo "[INFO] Vanilla Nanotron training done"

# Modified Nanotron training
cd ../src/nanotron-sp
pip install -e . > /dev/null 2>&1
cd - > /dev/null 2>&1
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=2 ../src/nanotron-sp/run_train.py --config-file config_tiny_llama_resume_sp.yaml > /dev/null 2>&1
echo "[INFO] Modified Nanotron training done"

# Evaluate the models
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=1  validate.py --checkpoint_1_path checkpoints/vanilla/100 --checkpoint_2_path checkpoints/sp/100
