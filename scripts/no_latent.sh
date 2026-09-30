# conda activate monet
# cd Monet

# SFT no latent loss
LATENT_SIZE=0
CE_EMPHASIZE_FACTOR=4.0
ALIGNMENT_WEIGHT=2.0
EMPHASIZE_LATENT_WEIGHT=2.0
SAVE_CKPT=sft_only_ce${CE_EMPHASIZE_FACTOR}
torchrun --nproc-per-node=8 --master-port=29501 -m src.main_omni \
  --epochs 2 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --stage "sft_stage2" \
  --data_path \
    "train_dataset/sft/filtered_finevideo_9k_no_latent_loss.json" \
  --load_model_path path/to/model \
  --save_model_path 8k_pure_ce_loss_finevideo_checkpoints/sft_stage1/${SAVE_CKPT} \
  --dataset_root path/to/dataset \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --allow_no_observation \
  --latent_size ${LATENT_SIZE} \
  --wandb_name ${SAVE_CKPT} \
  --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR}

  