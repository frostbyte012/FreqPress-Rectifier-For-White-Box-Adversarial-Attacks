#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

# DeiT-Base (timm), skip very slow AutoAttack, keep samples small
python -u -m attacks.comprehensive_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 32 \
  --num-workers 4 \
  --max-samples 500 \
  --no-run-cw \
  --no-run-deepfool \
  2>&1 | tee "logs/comprehensive_deit_base_fast_noCW_noDeepFool.log"

# DeiT-Base (timm), include DeepFool, skip C&W, skip AutoAttack
python -u -m attacks.comprehensive_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 16 \
  --num-workers 4 \
  --max-samples 200 \
  --no-run-cw \
  2>&1 | tee "logs/comprehensive_deit_base_noCW_withDeepFool.log"

# ResNet18 baseline (timm), include both DeepFool and C&W (can be slow)
python -u -m attacks.comprehensive_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "resnet18" \
  --batch-size 16 \
  --num-workers 4 \
  --max-samples 200 \
  2>&1 | tee "logs/comprehensive_resnet18_withCW_withDeepFool.log"

# DeiT-Base (timm), include AutoAttack (VERY slow) with tiny sample budget
python -u -m attacks.comprehensive_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 8 \
  --num-workers 4 \
  --max-samples 50 \
  --run-autoattack \
  --no-run-cw \
  --no-run-deepfool \
  2>&1 | tee "logs/comprehensive_deit_base_withAutoAttack_tiny.log"

# Optional: vary defense parameters
python -u -m attacks.comprehensive_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 16 \
  --num-workers 4 \
  --max-samples 200 \
  --no-run-cw \
  --no-run-deepfool \
  --cutoff 30 \
  --order 4 \
  --quality 50 \
  2>&1 | tee "logs/comprehensive_deit_base_fast_def_cut30_ord4_q50.log"
