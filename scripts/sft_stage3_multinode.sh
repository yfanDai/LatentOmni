#!/bin/bash

num_slots=8
HOSTFILE="${MPI_HOSTFILE:?Set MPI_HOSTFILE to your MPI hostfile}"
sed -i "s/slots=[0-9]\+\$/slots=$num_slots/g" "$HOSTFILE"

# 1. 设置 PATH
export PATH="${MPI_BIN_DIR:+${MPI_BIN_DIR}:}${PATH}"

# 2. 生成 .deepspeed_env 文件，将 LD_PRELOAD 写入其中
# 请确保把下面的路径替换为你实际的 libnccl.so.2 路径
echo "LD_PRELOAD=${NCCL_LIB_PATH:?Set NCCL_LIB_PATH to libnccl.so.2}" > .deepspeed_env

# (可选) 如果你希望 PATH 也传递给所有机器，可以一并写入
echo "PATH=$PATH" >> .deepspeed_env

# 3. 正常启动 DeepSpeed
deepspeed \
  --master_port 29666 \
  --no_local_rank \
  --hostfile "$HOSTFILE" \
  --launcher pdsh \
  --module src.main_omni \
  --epochs 2 \
  --bsz 1 \
  --grad_accum_steps 4 \
  --stage "sft_stage2" \
  --data_path "train_dataset/sft/filtered_all_8w_no_latent_loss.json" \
  --load_model_path path/to/model \
  --save_model_path 8w_pure_ce_loss_all_checkpoints/sft_stage1/sft_only_ce4.0 \
  --dataset_root path/to/dataset \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --allow_no_observation \
  --latent_size 0 \
  --ce_emphasize_factor 1.0 \
  --wandb_name sft_only_ce4.0
