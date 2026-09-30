#!/bin/bash

MODEL_PATH="path/to/model"
FILE_PATH="path/to/video.mp4"
OUTPUT="outputs/attention_heatmap_latent.png"

PROMPT="What is the primary purpose of the white car maneuvering through the water-filled obstacle course with orange cones?
A. Testing acceleration performance on wet surfaces
B. Conducting routine vehicle safety inspections
C. Demonstrating vehicle water resistance capabilities
D. Simulating emergency response driving scenarios"

SYS_PROMPT="You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

python pipeline/attention_heatmap_latent.py \
    --model_path   "${MODEL_PATH}" \
    --modality     video \
    --file_path    "${FILE_PATH}" \
    --prompt       "${PROMPT}" \
    --sys_prompt   "${SYS_PROMPT}" \
    --layer        -1 \
    --latent_size  40 \
    --max_new_tokens 256 \
    --max_plot_tokens 4000 \
    --use_audio_in_video \
    --temperature  0.4 \
    --output       "${OUTPUT}"
