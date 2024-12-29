#!/bin/bash -l
#
#SBATCH --job-name="nanotron_example"
#SBATCH --time=04:00:00
#SBATCH --partition=amdrtx
#SBATCH --nodelist=ault[43]
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --mem=200G
#SBATCH --output=nanotron_example.%j.o
#SBATCH --error=nanotron_example.%j.o
#SBATCH --account=g34


# Take command-line parameters
gpus=$1
config_id=$2

if [ -z "$gpus" ] || [ -z "$config_id" ]; then
  echo "Usage: $0 <gpus> <config_id>"
  exit 1
fi

#SBATCH --gpus-per-task=$gpus
config_dir="./wd_$config_id"
mkdir -p "$config_dir"
#SBATCH --output=${config_dir}/%j.o
#SBATCH --error=${config_dir}/%j.o

export NSYS_REPORT_DIR="$config_dir/nsys_reports"
rm -rf $NSYS_REPORT_DIR
mkdir -p $NSYS_REPORT_DIR

srun bash eval.sh ${gpus} "${config_dir}/conf.yaml"

# srun ~/opt/nvidia/nsight-systems-cli/2024.5.1/bin/nsys profile --trace=nvtx,cuda  --cuda-memory-usage=false --cuda-um-cpu-page-faults=false --cuda-um-gpu-page-faults=false -s none --output=${NSYS_REPORT_DIR}/nanotron_llama_train_nsys_report_%h_%p bash eval.sh ${gpus} "${config_dir}/conf.yaml"

return_code=$?

for report_file in ${NSYS_REPORT_DIR}/*.nsys-rep; do
  if [ -f "$report_file" ]; then
    sqlite_file="${report_file%.nsys-rep}.sqlite"
    ~/opt/nvidia/nsight-systems-cli/2024.5.1/bin/nsys export --type=sqlite --ts-normalize=true --output="$sqlite_file" "$report_file"
    echo "Exported $report_file to $sqlite_file"
  fi
done

exit $return_code
