#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

# DeiT-Base (timm), epsilon sweep
python -u -m attacks.pgd_evaluation \
	--data-dir "/path/to/imagenet-mini" \
	--model "deit_base_patch16_224" \
	--batch-size 64 \
	--num-workers 4 \
	--max-samples 500 \
	--epsilon 0.03 \
	--pgd-steps 10 \
	--apgd-steps 10 \
	--apgd-restarts 1 \
	2>&1 | tee "logs/pgd_apgd_deit_base_eps0.03_pgd10_apgd10_r1.log"

python -u -m attacks.pgd_evaluation \
	--data-dir "/path/to/imagenet-mini" \
	--model "deit_base_patch16_224" \
	--batch-size 64 \
	--num-workers 4 \
	--max-samples 500 \
	--epsilon 0.05 \
	--pgd-steps 10 \
	--apgd-steps 10 \
	--apgd-restarts 1 \
	2>&1 | tee "logs/pgd_apgd_deit_base_eps0.05_pgd10_apgd10_r1.log"

python -u -m attacks.pgd_evaluation \
	--data-dir "/path/to/imagenet-mini" \
	--model "deit_base_patch16_224" \
	--batch-size 64 \
	--num-workers 4 \
	--max-samples 500 \
	--epsilon 0.10 \
	--pgd-steps 10 \
	--apgd-steps 10 \
	--apgd-restarts 1 \
	2>&1 | tee "logs/pgd_apgd_deit_base_eps0.10_pgd10_apgd10_r1.log"

# DeiT-Base (timm), step/restart sweep
python -u -m attacks.pgd_evaluation \
	--data-dir "/path/to/imagenet-mini" \
	--model "deit_base_patch16_224" \
	--batch-size 32 \
	--num-workers 4 \
	--max-samples 200 \
	--epsilon 0.03 \
	--pgd-steps 40 \
	--apgd-steps 20 \
	--apgd-restarts 2 \
	2>&1 | tee "logs/pgd_apgd_deit_base_eps0.03_pgd40_apgd20_r2.log"

# ResNet18 baseline (timm), epsilon/steps
python -u -m attacks.pgd_evaluation \
	--data-dir "/path/to/imagenet-mini" \
	--model "resnet18" \
	--batch-size 128 \
	--num-workers 4 \
	--max-samples 1000 \
	--epsilon 0.03 \
	--pgd-steps 10 \
	--apgd-steps 10 \
	--apgd-restarts 1 \
	2>&1 | tee "logs/pgd_apgd_resnet18_eps0.03_pgd10_apgd10_r1.log"

python -u -m attacks.pgd_evaluation \
	--data-dir "/path/to/imagenet-mini" \
	--model "resnet18" \
	--batch-size 128 \
	--num-workers 4 \
	--max-samples 1000 \
	--epsilon 0.05 \
	--pgd-steps 40 \
	--apgd-steps 20 \
	--apgd-restarts 2 \
	2>&1 | tee "logs/pgd_apgd_resnet18_eps0.05_pgd40_apgd20_r2.log"

# Optional: stronger defense settings (freqpress cutoff/order/quality)
python -u -m attacks.pgd_evaluation \
	--data-dir "/path/to/imagenet-mini" \
	--model "deit_base_patch16_224" \
	--batch-size 64 \
	--num-workers 4 \
	--max-samples 200 \
	--epsilon 0.03 \
	--pgd-steps 10 \
	--apgd-steps 10 \
	--apgd-restarts 1 \
	--cutoff 30 \
	--order 4 \
	--quality 50 \
	2>&1 | tee "logs/pgd_apgd_deit_base_eps0.03_def_cut30_ord4_q50.log"

