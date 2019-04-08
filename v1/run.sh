#!/bin/bash

INPUT_DIR=$1
OUTPUT_FILE=$2

(python3 reconstruct.py --input_dir "${INPUT_DIR}" --id 0)&
(python3 reconstruct.py --input_dir "${INPUT_DIR}" --id 1)&
(python3 reconstruct.py --input_dir "${INPUT_DIR}" --id 2)&
(python3 reconstruct.py --input_dir "${INPUT_DIR}" --id 3)&
(python3 reconstruct.py --input_dir "${INPUT_DIR}" --id 4)&
(python3 reconstruct.py --input_dir "${INPUT_DIR}" --id 5)&
(python3 reconstruct.py --input_dir "${INPUT_DIR}" --id 6)&
(python3 reconstruct.py --input_dir "${INPUT_DIR}" --id 7)&
wait

python3 ./predict.py --output_file "${OUTPUT_FILE}"
rm -rf ./tmp/
