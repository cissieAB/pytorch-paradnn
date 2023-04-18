#!/bin/bash

## Read the csv files located at $INPUT_PATH, find the highest tflops by data type.
## Output the rows with highest tflops under current folder.
## Sample usage: sh find_max_tflops.sh ../../results max_flops.csv

INPUT_PATH=$1
OUTPUT_FILENAME=$2

# the same as that in fc.py
HEADER_LINE="device,input_type,layers,nodes,batch_size,input_size,output_size,#params,duration,tflops"

# Check the file is exists or not
if [ -f ${OUTPUT_FILENAME} ]; then
   rm ${OUTPUT_FILENAME}
   echo "${OUTPUT_FILENAME} is removed"
fi

echo $HEADER_LINE >> ${OUTPUT_FILENAME} # >> to write to the end

# sciml21* indicates T4, sciml23* indicates A100
for filename in "${INPUT_PATH}"/*; do
  echo "Reading data in " ${filename};
  python3 find_max_flops.py -i ${filename} >> ${OUTPUT_FILENAME};
done
