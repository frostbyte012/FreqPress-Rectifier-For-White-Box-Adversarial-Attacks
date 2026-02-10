#!/usr/bin/env bash
set -euo pipefail

# FGSM runs (explicit commands, no loops)
# NOTE: This module uses timm model identifiers via --model (NOT Patch-Fool's --network)

mkdir -p logs

# DeiT-Base (timm)
python -u -m attacks.fgsm_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 64 \
  --num-workers 4 \
  --max-samples 1000 \
  --epsilon 0.01 \
  2>&1 | tee "logs/fgsm_deit_base_eps0.01.log"

python -u -m attacks.fgsm_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 64 \
  --num-workers 4 \
  --max-samples 1000 \
  --epsilon 0.03 \
  2>&1 | tee "logs/fgsm_deit_base_eps0.03.log"

python -u -m attacks.fgsm_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 64 \
  --num-workers 4 \
  --max-samples 1000 \
  --epsilon 0.05 \
  2>&1 | tee "logs/fgsm_deit_base_eps0.05.log"

# ResNet18 baseline (timm)
python -u -m attacks.fgsm_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "resnet18" \
  --batch-size 128 \
  --num-workers 4 \
  --max-samples 2000 \
  --epsilon 0.01 \
  2>&1 | tee "logs/fgsm_resnet18_eps0.01.log"

python -u -m attacks.fgsm_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "resnet18" \
  --batch-size 128 \
  --num-workers 4 \
  --max-samples 2000 \
  --epsilon 0.03 \
  2>&1 | tee "logs/fgsm_resnet18_eps0.03.log"

python -u -m attacks.fgsm_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "resnet18" \
  --batch-size 128 \
  --num-workers 4 \
  --max-samples 2000 \
  --epsilon 0.05 \
  2>&1 | tee "logs/fgsm_resnet18_eps0.05.log"

# Optional: vary defense parameters (cutoff/order/quality)
python -u -m attacks.fgsm_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 64 \
  --num-workers 4 \
  --max-samples 500 \
  --epsilon 0.03 \
  --cutoff 30 \
  --order 4 \
  --quality 50 \
  2>&1 | tee "logs/fgsm_deit_base_eps0.03_def_cut30_ord4_q50.log"
