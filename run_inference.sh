#!/bin/bash

MODEL_NAME="HuggingFaceH4/zephyr-7b-beta"
CKPT_DIR="/home/atharva_inamdar/llm_inference/checkpoint"
DATA_PATH="/home/atharva_inamdar/llm_inference/infer_data.json"
OUTPUT_DIR="output"

accelerate launch inference.py \
${MODEL_NAME} \
${CKPT_DIR} \
${DATA_PATH} \
${OUTPUT_DIR}