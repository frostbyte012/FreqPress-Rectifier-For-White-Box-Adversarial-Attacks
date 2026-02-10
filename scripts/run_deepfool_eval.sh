#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

# DeiT-Base (timm), steps/overshoot sweep
python -u -m attacks.deepfool_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 16 \
  --num-workers 4 \
  --max-samples 200 \
  --steps 50 \
  --overshoot 0.02 \
  2>&1 | tee "logs/deepfool_deit_base_steps50_over0.02.log"

python -u -m attacks.deepfool_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 16 \
  --num-workers 4 \
  --max-samples 200 \
  --steps 100 \
  --overshoot 0.02 \
  2>&1 | tee "logs/deepfool_deit_base_steps100_over0.02.log"

python -u -m attacks.deepfool_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 16 \
  --num-workers 4 \
  --max-samples 200 \
  --steps 50 \
  --overshoot 0.10 \
  2>&1 | tee "logs/deepfool_deit_base_steps50_over0.10.log"

# ResNet18 baseline (timm)
python -u -m attacks.deepfool_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "resnet18" \
  --batch-size 32 \
  --num-workers 4 \
  --max-samples 500 \
  --steps 50 \
  --overshoot 0.02 \
  2>&1 | tee "logs/deepfool_resnet18_steps50_over0.02.log"

# Optional: vary defense parameters
python -u -m attacks.deepfool_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 16 \
  --num-workers 4 \
  --max-samples 200 \
  --steps 50 \
  --overshoot 0.02 \
  --cutoff 30 \
  --order 4 \
  --quality 50 \
  2>&1 | tee "logs/deepfool_deit_base_steps50_over0.02_def_cut30_ord4_q50.log"
