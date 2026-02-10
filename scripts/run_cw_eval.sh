#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

# DeiT-Base (timm), trade-off constant sweep
python -u -m attacks.cw_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 16 \
  --num-workers 4 \
  --max-samples 200 \
  --c 0.1 \
  --steps 50 \
  2>&1 | tee "logs/cw_deit_base_c0.1_steps50.log"

python -u -m attacks.cw_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 16 \
  --num-workers 4 \
  --max-samples 200 \
  --c 1.0 \
  --steps 100 \
  2>&1 | tee "logs/cw_deit_base_c1.0_steps100.log"

python -u -m attacks.cw_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 16 \
  --num-workers 4 \
  --max-samples 200 \
  --c 10.0 \
  --steps 100 \
  2>&1 | tee "logs/cw_deit_base_c10_steps100.log"

# ResNet18 baseline (timm)
python -u -m attacks.cw_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "resnet18" \
  --batch-size 32 \
  --num-workers 4 \
  --max-samples 500 \
  --c 1.0 \
  --steps 100 \
  2>&1 | tee "logs/cw_resnet18_c1.0_steps100.log"

# Optional: vary defense parameters
python -u -m attacks.cw_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 16 \
  --num-workers 4 \
  --max-samples 200 \
  --c 1.0 \
  --steps 100 \
  --cutoff 30 \
  --order 4 \
  --quality 50 \
  2>&1 | tee "logs/cw_deit_base_c1.0_steps100_def_cut30_ord4_q50.log"
