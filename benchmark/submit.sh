#!/bin/bash -l

# Take command-line parameters
gpus=$1
top_directory=$2

if [ -z "$gpus" ] || [ -z "$top_directory" ]; then
  echo "Usage: $0 <gpus> <top_directory>"
  exit 1
fi
mkdir -p "$top_directory"

#
#SBATCH --job-name="nanotron_train"
#SBATCH --time=04:00:00
#SBATCH --partition=amdrtx
#SBATCH --nodelist=ault[43]
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=$gpus
#SBATCH --mem=200G
#SBATCH --output=${top_directory}/%j.o
#SBATCH --error=${top_directory}/%j.o
#SBATCH --account=g34


export NSYS_REPORT_DIR="$top_directory/nsys_reports"
rm -rf $NSYS_REPORT_DIR
mkdir -p $NSYS_REPORT_DIR

srun bash eval.sh ${gpus} "${top_directory}/conf.yaml"

# srun ~/opt/nvidia/nsight-systems-cli/2024.5.1/bin/nsys profile --trace=nvtx,cuda  --cuda-memory-usage=false --cuda-um-cpu-page-faults=false --cuda-um-gpu-page-faults=false -s none --output=${NSYS_REPORT_DIR}/nanotron_llama_train_nsys_report_%h_%p bash eval.sh ${gpus} "${top_directory}/conf.yaml"

return_code=$?

# for report_file in ${NSYS_REPORT_DIR}/*.nsys-rep; do
#   if [ -f "$report_file" ]; then
#     sqlite_file="${report_file%.nsys-rep}.sqlite"
#     ~/opt/nvidia/nsight-systems-cli/2024.5.1/bin/nsys export --type=sqlite --ts-normalize=true --output="$sqlite_file" "$report_file"
#     echo "Exported $report_file to $sqlite_file"
#   fi
# done

exit $return_code
