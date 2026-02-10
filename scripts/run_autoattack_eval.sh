#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

# Linf, DeiT-Base (timm) (AutoAttack is slow; keep max-samples small)
python -u -m attacks.autoattack_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 8 \
  --num-workers 4 \
  --max-samples 100 \
  --norm Linf \
  --version standard \
  --epsilon 0.03 \
  2>&1 | tee "logs/autoattack_deit_base_normLinf_eps0.03_standard.log"

python -u -m attacks.autoattack_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 8 \
  --num-workers 4 \
  --max-samples 100 \
  --norm Linf \
  --version standard \
  --epsilon 0.05 \
  2>&1 | tee "logs/autoattack_deit_base_normLinf_eps0.05_standard.log"

# Linf, randomized version
python -u -m attacks.autoattack_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 8 \
  --num-workers 4 \
  --max-samples 100 \
  --norm Linf \
  --version rand \
  --epsilon 0.03 \
  2>&1 | tee "logs/autoattack_deit_base_normLinf_eps0.03_rand.log"

# L2, DeiT-Base (timm)
python -u -m attacks.autoattack_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 8 \
  --num-workers 4 \
  --max-samples 100 \
  --norm L2 \
  --version standard \
  --epsilon 0.5 \
  2>&1 | tee "logs/autoattack_deit_base_normL2_eps0.5_standard.log"

# ResNet18 baseline (timm)
python -u -m attacks.autoattack_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "resnet18" \
  --batch-size 16 \
  --num-workers 4 \
  --max-samples 200 \
  --norm Linf \
  --version standard \
  --epsilon 0.03 \
  2>&1 | tee "logs/autoattack_resnet18_normLinf_eps0.03_standard.log"

# Optional: vary defense parameters
python -u -m attacks.autoattack_evaluation \
  --data-dir "/path/to/imagenet-mini" \
  --model "deit_base_patch16_224" \
  --batch-size 8 \
  --num-workers 4 \
  --max-samples 100 \
  --norm Linf \
  --version standard \
  --epsilon 0.03 \
  --cutoff 30 \
  --order 4 \
  --quality 50 \
  2>&1 | tee "logs/autoattack_deit_base_normLinf_eps0.03_def_cut30_ord4_q50.log"
