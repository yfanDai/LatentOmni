#!/bin/bash

MODEL_PATH="path/to/model"
FILE_PATH="path/to/video.mp4"
OUTPUT="outputs/attention_heatmap_ce.png"

PROMPT="Given the combination of visual text and audio content, what is the likely theme or topic being explored?\nA. A sports analysis with background singing.\nB. A scientific lecture on quantum physics illustrated with vocal performances.\nC. A cooking tutorial accompanied by classical music.\nD. A lighthearted pop song with abstract visual elements."

SYS_PROMPT="You are a helpful AVQA expert."

python attention_line_latent.py \
    --model_path   "${MODEL_PATH}" \
    --modality     video \
    --file_path    "${FILE_PATH}" \
    --prompt       "${PROMPT}" \
    --sys_prompt   "${SYS_PROMPT}" \
    --layer        15 \
    --latent_size  0 \
    --max_new_tokens 256 \
    --max_plot_tokens 4000 \
    --use_audio_in_video \
    --temperature  0.4 \
    --output       "${OUTPUT}"
