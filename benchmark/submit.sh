#!/bin/bash

# Take command-line parameters
gpus=$1
top_directory=$2

export NSYS_REPORT_DIR="./${top_directory}/nsys_reports"
rm -rf $NSYS_REPORT_DIR
mkdir -p $NSYS_REPORT_DIR

return_code=$?

~/opt/nvidia/nsight-systems-cli/2024.5.1/bin/nsys profile --trace=nvtx,cuda  --cuda-memory-usage=false --cuda-um-cpu-page-faults=false --cuda-um-gpu-page-faults=false -s none --output=${NSYS_REPORT_DIR}/nanotron_llama_train_nsys_report_%h_%p bash eval.sh ${gpus} "${top_directory}/conf.yaml"

for report_file in ${NSYS_REPORT_DIR}/*.nsys-rep; do
  if [ -f "$report_file" ]; then
    sqlite_file="${report_file%.nsys-rep}.sqlite"
    ~/opt/nvidia/nsight-systems-cli/2024.5.1/bin/nsys export --type=sqlite --ts-normalize=true --output="$sqlite_file" "$report_file"
    echo "Exported $report_file to $sqlite_file"
  fi
done

exit $return_code