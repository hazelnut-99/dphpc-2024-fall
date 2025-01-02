#!/bin/bash

#
#SBATCH --job-name="usp_additional"
#SBATCH --time=04:00:00
#SBATCH --partition=amdrtx
#SBATCH --nodelist=ault[41]
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --mem=200G
#SBATCH --output=usp_benchmark_%j.o
#SBATCH --error=usp_benchmark_%j.e
#SBATCH --account=g34


# Check if TOP_DIR is provided as a command-line argument
if [ -z "$1" ]; then
    echo "Error: TOP_DIR must be provided as a command-line argument."
    echo "Usage: $0 <TOP_DIR>"
    exit 1
fi

echo "hello world"

# Set up environment
shopt -s expand_aliases
source ~/.bashrc
env_init
rm -rf checkpoints
export HUGGINGFACE_TOKEN=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
export CUDA_DEVICE_MAX_CONNECTIONS=1
export LD_PRELOAD=/users/zhu/nccl_nvtx_npkit_v2.20.5-1/nccl/build/lib/libnccl.so

cd ../src/nanotron-sp
pip install -e . > /dev/null 2>&1
cd - > /dev/null 2>&1

which torchrun

TOP_DIR=$1

echo "Processing top directory: $TOP_DIR"

NSYS_PATH=~/opt/nvidia/nsight-systems-cli/2024.5.1/bin/nsys

# Loop through each subdirectory in TOP_DIR
for subdir in "$TOP_DIR"/*; do
    if [ -d "$subdir" ]; then
        echo "Processing directory: $subdir"

        # Check if the return_code file already exists
        if [ -f "$subdir/return_code" ]; then
            echo "Return code file already exists for $subdir. Skipping this directory."
            continue
        fi

        # Check if gpus file exists and read the number of GPUs
        if [ -f "$subdir/per_node_gpus" ]; then
            GPUS=$(cat "$subdir/per_node_gpus")
        else
            echo "Warning: No 'gpus' file found in $subdir. Skipping this directory."
            continue
        fi

        # Run the nsys profile command
        srun $NSYS_PATH profile --trace=nvtx,cuda \
          --cuda-memory-usage=false --cuda-um-cpu-page-faults=false --cuda-um-gpu-page-faults=false -s none \
          --output="${subdir}/nanotron_nsys_report_%h_%p" \
          torchrun --nproc_per_node=$GPUS ../src/nanotron-sp/run_train.py \
          --config-file "$subdir/conf.yaml" > "$subdir/stdout.o" 2> "$subdir/stderr.e"

        # Capture the return code and log it
        RETURN_CODE=$?
        echo $RETURN_CODE >> "$subdir/return_code"

    fi
done
