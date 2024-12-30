#!/bin/bash


TOP_DIR=$1

if [ -z "$TOP_DIR" ]; then
    echo "Error: TOP_DIR must be provided as a command-line argument."
    echo "Usage: $0 <TOP_DIR>"
    exit 1
fi


for subdir in "$TOP_DIR"/*; do
    if [ -d "$subdir" ]; then
        echo "Processing directory: $subdir"

        for report_file in "$subdir"/*.nsys-rep; do
            if [ -f "$report_file" ]; then
                sqlite_file="${report_file%.nsys-rep}.sqlite"
                ~/opt/nvidia/nsight-systems-cli/2024.5.1/bin/nsys export --type=sqlite --ts-normalize=true --output="$sqlite_file" "$report_file"
                echo "Exported $report_file to $sqlite_file"
            fi
        done
    fi
done
