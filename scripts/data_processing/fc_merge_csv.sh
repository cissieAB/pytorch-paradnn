#!/bin/bash

# Read the csv files at $INPUT_PATH and merge them into a big one $OUTPUT_FILE_PATH
# Sample usage: sh scripts/data_processing/fc_merge_csv.sh res_2023/Feb_13 res_2023/fc_T4_f32.csv

INPUT_PATH=$1
OUTPUT_FILE_PATH=$2

# the same as that in fc.py
HEADER_LINE="device,input_type,layers,nodes,batch_size,input_size,output_size,#params,duration,tflops"

echo $HEADER_LINE > ${OUTPUT_FILE_PATH}

# sciml21* indicates T4, sciml23* indicates A100
for filename in ${INPUT_PATH}/sciml21*/*/benchmark_*.csv; do
  echo "Reading input from " $filename
  while IFS=, read -r line; do
    if [[ $line == ${HEADER_LINE} ]]; then
      continue
    fi
    echo $line >> ${OUTPUT_FILE_PATH}  # >> to write to the end
  done < $filename
done;
