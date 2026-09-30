MODEL_PATH="path/to/model"
FILE_PATH="path/to/video.mp4"

PROMPT="What visual detail was present when the audio mentioned discovering a blue tag in the forest?\nA. The man touching fungus while estimating its age\nB. The man sitting on a log examining fungus\nC. The man wearing a feathered hat and holding a pendant\nD. The man wearing a navy jacket and speaking to the camera."

python attention_video_overlay_latent.py \
    --model_path   "${MODEL_PATH}" \
    --file_path    "${FILE_PATH}" \
    --prompt       "${PROMPT}" \
    --sys_prompt "You are an AVQA expert." \
    --output_dir outputs/latent_av_vis \
    --top_k_frames 5 \
    --latent_size 40 \
    --layer 13 \
    --skip_overlay_video
