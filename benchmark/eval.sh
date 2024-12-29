#!/bin/bash

# Check if required parameters are provided
gpus=$1
config_file=$2

if [ -z "$gpus" ] || [ -z "$config_file" ]; then
  echo "Usage: $0 <gpus> <config_file>"
  exit 1
fi

echo $gpus
echo $config_file

# Init environment
shopt -s expand_aliases
source ~/.bashrc
env_init
rm -rf checkpoints
export HUGGINGFACE_TOKEN=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

# Modified Nanotron training
cd ../src/nanotron-sp
pip install -e . > /dev/null 2>&1
cd - > /dev/null 2>&1
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=$gpus ../src/nanotron-sp/run_train.py --config-file $config_file
