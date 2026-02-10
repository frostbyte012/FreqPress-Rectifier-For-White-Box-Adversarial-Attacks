#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

# DeiT-Base, 1 patch, no defense
python -u -m attacks.patch_fool \
	--network DeiT-B \
	--dataset ImageNet \
	--data_dir "/path/to/imagenet-mini" \
	--batch_size 64 \
	--workers 4 \
	--dataset_size 1 \
	--num_patch 1 \
	--patch_select Attn \
	--defense none \
	2>&1 | tee "logs/patchfool_DeiT-B_np1_Attn_def-none.log"

# DeiT-Base, 1 patch, FreqPress defense
python -u -m attacks.patch_fool \
	--network DeiT-B \
	--dataset ImageNet \
	--data_dir "/path/to/imagenet-mini" \
	--batch_size 64 \
	--workers 4 \
	--dataset_size 1 \
	--num_patch 1 \
	--patch_select Attn \
	--defense butter_webp \
	2>&1 | tee "logs/patchfool_DeiT-B_np1_Attn_def-butter_webp.log"

# DeiT-Base, 2 patches, attention selection, no defense
python -u -m attacks.patch_fool \
	--network DeiT-B \
	--dataset ImageNet \
	--data_dir "/path/to/imagenet-mini" \
	--batch_size 64 \
	--workers 4 \
	--dataset_size 1 \
	--num_patch 2 \
	--patch_select Attn \
	--defense none \
	2>&1 | tee "logs/patchfool_DeiT-B_np2_Attn_def-none.log"

# DeiT-Base, 1 patch, saliency selection, no defense
python -u -m attacks.patch_fool \
	--network DeiT-B \
	--dataset ImageNet \
	--data_dir "/path/to/imagenet-mini" \
	--batch_size 64 \
	--workers 4 \
	--dataset_size 1 \
	--num_patch 1 \
	--patch_select Saliency \
	--defense none \
	2>&1 | tee "logs/patchfool_DeiT-B_np1_Saliency_def-none.log"

# DeiT-Base, mild L-inf constraint (0-1 range), no defense
python -u -m attacks.patch_fool \
	--network DeiT-B \
	--dataset ImageNet \
	--data_dir "/path/to/imagenet-mini" \
	--batch_size 64 \
	--workers 4 \
	--dataset_size 1 \
	--num_patch 1 \
	--patch_select Attn \
	--mild_l_inf 0.03 \
	--defense none \
	2>&1 | tee "logs/patchfool_DeiT-B_np1_Attn_linf0.03_def-none.log"

# DeiT-Base, mild L2 constraint (0-16 range), no defense
python -u -m attacks.patch_fool \
	--network DeiT-B \
	--dataset ImageNet \
	--data_dir "/path/to/imagenet-mini" \
	--batch_size 64 \
	--workers 4 \
	--dataset_size 1 \
	--num_patch 1 \
	--patch_select Attn \
	--mild_l_2 4 \
	--defense none \
	2>&1 | tee "logs/patchfool_DeiT-B_np1_Attn_l2_4_def-none.log"

# ResNet50, 1 patch, no defense
python -u -m attacks.patch_fool \
	--network ResNet50 \
	--dataset ImageNet \
	--data_dir "/path/to/imagenet-mini" \
	--batch_size 64 \
	--workers 4 \
	--dataset_size 1 \
	--num_patch 1 \
	--patch_select Saliency \
	--defense none \
	2>&1 | tee "logs/patchfool_ResNet50_np1_Saliency_def-none.log"

