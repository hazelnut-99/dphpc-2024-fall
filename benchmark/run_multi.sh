#!/bin/bash -l
#
#SBATCH --job-name="usp_multi_benchmark"
#SBATCH --time=04:00:00
#SBATCH --partition=amdrtx
#SBATCH --nodelist=ault[43-44]
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --mem=200G
#SBATCH --output=nanotron_example.%j.o
#SBATCH --error=nanotron_example.%j.e
#SBATCH --account=g34

conda activate base
conda activate nanotron-sp

module load openmpi/4.1.1
module load cuda/12.1.1

srun nvidia-smi -L


export LD_PRELOAD=/users/zhu/nccl_nvtx_npkit_v2.20.5-1/nccl/build/lib/libnccl.so
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HUGGINGFACE_TOKEN=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
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

TOP_DIR=$1
echo "Processing top directory: $TOP_DIR"

export OMP_NUM_THREADS=16  ## Unused
NNODES=$SLURM_NNODES

# define the node 0 hostname:port
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)  ## use hostname not ip
MASTER_PORT=29500
echo "Master addr: $MASTER_ADDR:$MASTER_PORT"


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

        LAUNCHER="python -u -m torch.distributed.run \
            --nproc_per_node $GPUS \
            --nnodes $NNODES \
            --node_rank $SLURM_PROCID \
            --rdzv_id $SLURM_JOB_ID \
            --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
            --rdzv_backend c10d \
            --max_restarts 0 \
            --role $(hostname -s|tr -dc '0-9'): "

        # Check that relative paths to your `run_train.py` are correct
        PROGRAM="--master_port $MASTER_PORT ../src/nanotron-sp/run_train.py --config-file $subdir/conf.yaml > $subdir/stdout.o 2> $subdir/stderr.e"

        export CMD="${LAUNCHER} ${PROGRAM}"

        echo $CMD


        srun ~/opt/nvidia/nsight-systems-cli/2024.5.1/bin/nsys profile --trace=nvtx,cuda \
         --cuda-memory-usage=false --cuda-um-cpu-page-faults=false \
         --cuda-um-gpu-page-faults=false -s none \
          --output="${subdir}/nanotron_nsys_report_%h_%p" bash -c "$CMD"

        # Capture the return code and log it
        RETURN_CODE=$?
        echo $RETURN_CODE >> "$subdir/return_code"
    fi
done








