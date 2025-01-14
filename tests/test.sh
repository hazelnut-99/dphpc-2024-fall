#!/bin/bash

# Init environment
shopt -s expand_aliases
source ~/.bashrc
env_init
rm -rf checkpoints
rm -rf log_vanilla.txt log_sp.txt
mkdir -p checkpoints

# Generate the initial weights randomly and train the model for 10 steps
cd ../src/nanotron-vanilla
pip install -e . > /dev/null 2>&1
cd - > /dev/null 2>&1

CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=2 ../src/nanotron-vanilla/run_train.py --config-file config_tiny_llama_init.yaml > /dev/null 2>&1
echo "[INFO] Initial weights generated"

# Copy the initial weights to the resume directory for both the vanilla and the modified model
cp -r checkpoints/init checkpoints/vanilla
cp -r checkpoints/init checkpoints/sp

# Vanilla Nanotron training
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=1 ../src/nanotron-vanilla/run_train.py --config-file config_tiny_llama_resume_vanilla.yaml &> log_vanilla.txt
echo "[INFO] Vanilla Nanotron training done"

# Modified Nanotron training
cd ../src/nanotron-sp
pip install -e . > /dev/null 2>&1
cd - > /dev/null 2>&1
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 ../src/nanotron-vanilla/run_train.py --config-file config_tiny_llama_resume_sp.yaml &> log_sp.txt
echo "[INFO] Modified Nanotron training done"

# # Evaluate the models
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=1 validate.py --ring_ranks 2 --ulysses_ranks 2
