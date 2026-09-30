#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_DIR:?Set SOURCE_DIR to the source directory or rsync URL}"
: "${DEST_DIR:?Set DEST_DIR to the destination directory}"

rsync -avzh --progress \
  --exclude='checkpoints/' \
  --exclude='tensorboard_log/' \
  --exclude='wandb/' \
  "$SOURCE_DIR" "$DEST_DIR"
