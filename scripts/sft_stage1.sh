# conda activate monet
# cd Monet

# SFT stage1
LATENT_SIZE=8
CE_EMPHASIZE_FACTOR=4.0
ALIGNMENT_WEIGHT=0.5
EMPHASIZE_LATENT_WEIGHT=2.0
SAVE_CKPT=sft_stage1_ce${CE_EMPHASIZE_FACTOR}
torchrun --nproc-per-node=1 --master-port=29501 -m src.main_omni \
  --epochs 1 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --stage "sft_stage1" \
  --data_path \
    "train_dataset/sft/video_num_filtered_3k_test_sft_with_latent_loss.json" \
  --load_model_path path/to/model \
  --save_model_path 8_latent_ce_and_mse_loss_checkpoints/sft_stage1/${SAVE_CKPT} \
  --dataset_root path/to/dataset \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --allow_no_observation \
  --latent_size ${LATENT_SIZE} \
  --wandb_name ${SAVE_CKPT} \
  --ce_emphasize_factor ${CE_EMPHASIZE_FACTOR} \
  --alignment_weight ${ALIGNMENT_WEIGHT} \

  