#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

# DeiT-Base (timm), epsilon/steps sweep
python -u -m attacks.ifgsm_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 64 \
  --num-workers 4 \
  --max-samples 500 \
  --epsilon 0.01 \
  --steps 10 \
  2>&1 | tee "logs/ifgsm_deit_base_eps0.01_steps10.log"

python -u -m attacks.ifgsm_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 64 \
  --num-workers 4 \
  --max-samples 500 \
  --epsilon 0.03 \
  --steps 10 \
  2>&1 | tee "logs/ifgsm_deit_base_eps0.03_steps10.log"

python -u -m attacks.ifgsm_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 64 \
  --num-workers 4 \
  --max-samples 500 \
  --epsilon 0.03 \
  --steps 20 \
  2>&1 | tee "logs/ifgsm_deit_base_eps0.03_steps20.log"

python -u -m attacks.ifgsm_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 64 \
  --num-workers 4 \
  --max-samples 500 \
  --epsilon 0.05 \
  --steps 20 \
  2>&1 | tee "logs/ifgsm_deit_base_eps0.05_steps20.log"

# ResNet18 baseline (timm)
python -u -m attacks.ifgsm_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "resnet18" \
  --batch-size 128 \
  --num-workers 4 \
  --max-samples 1000 \
  --epsilon 0.03 \
  --steps 10 \
  2>&1 | tee "logs/ifgsm_resnet18_eps0.03_steps10.log"

python -u -m attacks.ifgsm_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "resnet18" \
  --batch-size 128 \
  --num-workers 4 \
  --max-samples 1000 \
  --epsilon 0.05 \
  --steps 20 \
  2>&1 | tee "logs/ifgsm_resnet18_eps0.05_steps20.log"

# Optional: vary defense parameters
python -u -m attacks.ifgsm_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 64 \
  --num-workers 4 \
  --max-samples 300 \
  --epsilon 0.03 \
  --steps 10 \
  --cutoff 30 \
  --order 4 \
  --quality 50 \
  2>&1 | tee "logs/ifgsm_deit_base_eps0.03_steps10_def_cut30_ord4_q50.log"
