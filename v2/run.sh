#!/bin/bash

INPUT_DIR=$1
OUTPUT_FILE=$2

python3 ./predict.py --input_dir "${INPUT_DIR}" --output_file "${OUTPUT_FILE}"
