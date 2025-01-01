#!/bin/bash -l
#
#SBATCH --job-name="nanotron_sp_multinode"
#SBATCH --time=04:00:00
#SBATCH --partition=amdrtx
#SBATCH --nodelist=ault[41-42]
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --mem=200G
#SBATCH --output=log/nanotron_multinode.%j.o
#SBATCH --error=log/nanotron_multinode.%j.e
#SBATCH --account=g34

# Init environment
shopt -s expand_aliases
source ~/.bashrc
env_init
module load openmpi/4.1.1
rm -rf checkpoints
mkdir -p checkpoints

srun nvidia-smi -L

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
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=1 ../src/nanotron-vanilla/run_train.py --config-file config_tiny_llama_resume_vanilla.yaml
echo "[INFO] Vanilla Nanotron training done"

# Modified Nanotron training
cd ../src/nanotron-sp
pip install -e . > /dev/null 2>&1
cd - > /dev/null 2>&1

export CUDA_DEVICE_MAX_CONNECTIONS=1 # Important for Nanotron
export OMP_NUM_THREADS=16  ## Unused
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

GPUS_PER_NODE=2  # EDIT if it's not 2-gpus per node
NNODES=$SLURM_NNODES

# define the node 0 hostname:port
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)  ## use hostname not ip
MASTER_PORT=29500
echo "Master addr: $MASTER_ADDR:$MASTER_PORT"

LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $SLURM_PROCID \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role $(hostname -s|tr -dc '0-9'): "

# Check that relative paths to your `run_train.py` are correct
PROGRAM="--master_port $MASTER_PORT ../src/nanotron-sp/run_train.py --config-file config_tiny_llama_resume_sp_multinode.yaml"

export CMD="${LAUNCHER} ${PROGRAM}"

echo $CMD

# bash -c is needed for the delayed interpolation of env vars to work
srun bash -c "$CMD"
echo "[INFO] Modified Nanotron training done"

# Evaluate the outputs
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=1 validate.py --ring_ranks 2 --ulysses_ranks 2
 