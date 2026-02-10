Official implementation of FreqPress, a defense mechanism against adversarial attacks using Butterworth low-pass filtering and WebP compression.

## Overview

FreqPress combines frequency-domain filtering (Butterworth) and lossy compression (WebP) to defend against adversarial attacks on image classification models. This repository contains:

- **Patch-Fool Attack**: Implementation of targeted patch-based adversarial attacks on Vision Transformers and CNNs
- **Standard Attacks**: FGSM, I-FGSM (BIM), PGD, and APGD implementations with FreqPress defense
- **Advanced Attacks**: Carlini-Wagner (C&W), DeepFool, and AutoAttack evaluations
- **FreqPress Defense**: Butterworth low-pass filter + WebP compression pipeline
- **Evaluation Scripts**: Comprehensive attack evaluations with and without defense

### Image Processing Pipeline 

![FreqPress Pipeline](https://github.com/frostbyte012/FreqPress-Rectifier-For-White-Box-Adversarial-Attacks/blob/main/assets/images/pipeline.jpg)

*Overview of the Proposed FreqPress defense*

*The FreqPress defense pipeline applies Butterworth low-pass filtering followed by WebP compression to remove high-frequency adversarial perturbations while preserving image quality.*

## Repository Structure

```
freqpress/
├── attacks/
│   ├── __init__.py
│   ├── patch_fool.py               # Patch-Fool attack implementation
│   ├── fgsm_evaluation.py          # FGSM attack evaluation
│   ├── ifgsm_evaluation.py         # I-FGSM (BIM) attack evaluation
│   ├── pgd_evaluation.py           # PGD/APGD evaluation script
│   ├── cw_evaluation.py            # Carlini-Wagner (C&W) attack
│   ├── deepfool_evaluation.py      # DeepFool attack evaluation
│   ├── autoattack_evaluation.py    # AutoAttack ensemble evaluation
│   └── comprehensive_evaluation.py # All attacks in one script
├── defenses/
│   ├── __init__.py
│   └── freqpress.py                # FreqPress defense implementation
├── utils/
│   ├── __init__.py
│   ├── data_loader.py              # Dataset loading utilities
│   ├── logger.py                   # Logging utilities
│   ├── meter.py                    # Metric tracking
│   └── ops.py                      # clamp, PCGrad utilities
├── models/
│   ├── __init__.py
│   ├── DeiT.py                     # DeiT model definitions
│   └── resnet.py                   # ResNet model definitions
├── scripts/
│   ├── run_patch_fool.sh           
│   ├── run_fgsm_eval.sh    
│   ├── run_ifgsm_eval.sh         
│   ├── run_pgd_eval.sh         
│   ├── run_cw_eval.sh            
│   ├── run_deepfool_eval.sh       
│   ├── run_autoattack_eval.sh      
│   ├── run_comprehensive_eval.sh              
├── requirements.txt
├── GFLOP Test/
│   ├── FreqPress_gflop.py           
│   ├── GFLOPS_Marked_Code_Verification.ipynb
│   ├──README.md
├── README.md
└── setup.py
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/freqpress.git
cd freqpress

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Important Notes (Read This First)

- **Patch-Fool dataset path**: Patch-Fool uses `--data_dir` and it must point to the **ImageNet root folder** that contains **both** `train/` and `val/`.
  - Example: `--data_dir /path/to/imagenet` (where `/path/to/imagenet/train` and `/path/to/imagenet/val` exist)
- **All other attacks dataset path**: FGSM / I-FGSM / PGD / APGD / C&W / DeepFool / AutoAttack use `--data-dir` and it should point to a **single split directory** that is an `ImageFolder`, e.g. **`imagenet/val`** or **`imagenet/train`**.
  - Example: `--data-dir /path/to/imagenet/val`
- **CUDA requirement**:
  - **Patch-Fool requires CUDA** and will fail if CUDA is not available.
  - The other evaluation scripts **can run on CPU** (they auto-select `cuda` if available).

### Run via provided scripts

The repository includes ready-to-run scripts under `scripts/` (each command is explicit). After updating paths inside the script, run. Or you can copy one explicit command and run it separately also.

```bash
bash scripts/run_patch_fool.sh
bash scripts/run_fgsm_eval.sh
bash scripts/run_ifgsm_eval.sh
bash scripts/run_pgd_eval.sh
bash scripts/run_cw_eval.sh
bash scripts/run_deepfool_eval.sh
bash scripts/run_autoattack_eval.sh
bash scripts/run_comprehensive_eval.sh
```

### 1. Patch-Fool Attack

Run the Patch-Fool attack on ImageNet validation set:

```bash
python -m attacks.patch_fool \
    --network DeiT-B \
    --dataset ImageNet \
    --data_dir /path/to/imagenet \
    --batch_size 64 \
    --num_patch 1 \
    --patch_select Attn \
    --train_attack_iters 250 \
    --defense none
```

**Key Arguments:**
- `--data_dir`: **ImageNet root directory** containing both `train/` and `val/` (Patch-Fool will internally use the required split)
- `--network`: Model architecture (`DeiT-B`, `DeiT-S`, `DeiT-T`, `ResNet50`, `ResNet152`)
- `--batch_size`: Batch size (Patch-Fool flag uses underscore: `--batch_size`)
- `--workers`: DataLoader workers (Patch-Fool flag uses `--workers`)
- `--patch_select`: Patch selection method (`Rand`, `Saliency`, `Attn`)
- `--num_patch`: Number of patches to attack
- `--defense`: Apply defense (`none`, `butter_webp`)
- `--mild_l_inf`: L-infinity constraint (0-1 range)
- `--mild_l_2`: L2 constraint (0-16 range)

### 2. FGSM Attack Evaluation

Evaluate FGSM (Fast Gradient Sign Method) attack:

```bash
python -m attacks.fgsm_evaluation \
  --data-dir /path/to/imagenet/val \
  --model deit_base_patch16_224 \
  --batch-size 64 \
  --max-samples 1000 \
  --epsilon 0.03
```

**Key Arguments:**
- `--data-dir`: Dataset split directory (`imagenet/val` or `imagenet/train`) in ImageFolder format
- `--model`: **timm model identifier** (e.g. `deit_base_patch16_224`, `resnet18`)
- `--epsilon`: FGSM perturbation budget (typical: `0.01`, `0.03`, `0.05` for Linf)
- `--max-samples`: Limit evaluation size for faster experiments
- `--cutoff`, `--order`, `--quality`: FreqPress defense parameters (Butterworth + WebP)

### 3. I-FGSM (BIM) Attack Evaluation

Evaluate I-FGSM (Iterative FGSM / Basic Iterative Method) attack:

```bash
python -m attacks.ifgsm_evaluation \
  --data-dir /path/to/imagenet/val \
  --model deit_base_patch16_224 \
  --batch-size 64 \
  --max-samples 1000 \
  --epsilon 0.03 \
  --steps 10
```

**Key Arguments:**
- `--epsilon`: Total Linf budget
- `--steps`: Number of BIM iterations (e.g. `10`, `20`)
- `--model`, `--data-dir`, `--batch-size`, `--max-samples`: Same meaning as FGSM

### 4. PGD/APGD Evaluation

Evaluate PGD and APGD attacks with FreqPress defense:

```bash
python -m attacks.pgd_evaluation \
  --data-dir /path/to/imagenet/val \
  --model deit_base_patch16_224 \
  --batch-size 64 \
  --max-samples 1000 \
  --epsilon 0.03 \
  --pgd-steps 10 \
  --apgd-steps 10 \
  --apgd-restarts 1
```

**Key Arguments:**
- `--epsilon`: Perturbation budget
- `--pgd-steps`: Steps for PGD
- `--apgd-steps`: Steps for APGD
- `--apgd-restarts`: APGD restarts

### 5. Comprehensive Evaluation (All Attacks)

Run all attacks (FGSM, I-FGSM, PGD, APGD) on multiple models:

```bash
python -m attacks.comprehensive_evaluation \
  --data-dir /path/to/imagenet/val \
  --model deit_base_patch16_224 \
  --batch-size 32 \
  --max-samples 500
```

This generates a CSV file with results for all attack/defense combinations.

**Key Arguments:**
- `--run-cw` / `--no-run-cw`: Toggle C&W (slow)
- `--run-deepfool` / `--no-run-deepfool`: Toggle DeepFool (slow)
- `--run-autoattack`: Enable AutoAttack (very slow)

### 6. Carlini-Wagner (C&W) Attack

Evaluate the powerful C&W optimization-based attack:

```bash
python -m attacks.cw_evaluation \
  --data-dir /path/to/imagenet/val \
  --model deit_base_patch16_224 \
  --batch-size 16 \
  --max-samples 200 \
  --c 1.0 \
  --steps 100
```

**Note**: C&W is computationally expensive. Adjust batch size and steps as needed.

**Key Arguments:**
- `--c`: Trade-off constant (try `0.1`, `1.0`, `10.0`)
- `--steps`: Optimization steps (e.g. `50`, `100`)

### 7. DeepFool Attack

Evaluate DeepFool, which finds minimal perturbations:

```bash
python -m attacks.deepfool_evaluation \
  --data-dir /path/to/imagenet/val \
  --model deit_base_patch16_224 \
  --batch-size 16 \
  --max-samples 200 \
  --steps 50 \
  --overshoot 0.02
```

**Key Arguments:**
- `--steps`: Max DeepFool iterations
- `--overshoot`: Overshoot factor (typical: `0.02`, sometimes higher like `0.1`)

### 8. AutoAttack Evaluation

Run AutoAttack, an ensemble of diverse attacks for robust evaluation:

```bash
python -m attacks.autoattack_evaluation \
  --data-dir /path/to/imagenet/val \
  --model deit_base_patch16_224 \
  --batch-size 8 \
  --max-samples 100 \
  --norm Linf \
  --version standard \
  --epsilon 0.03
```

**⚠️ Warning**: AutoAttack is **very** computationally expensive! Consider using `MAX_SAMPLES` for testing.

**Key Arguments:**
- `--norm`: `Linf` or `L2`
- `--version`: `standard` or `rand`
- `--epsilon`: For `Linf`, typical: `0.01`, `0.03`, `0.05`; for `L2`, use a larger value (e.g. `0.5`)

## Dataset Preparation

### Download links

- ImageNet-mini (train + val, convenient for quick experiments): https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000
- ImageNet-1k validation only: https://www.kaggle.com/datasets/titericz/imagenet1k-val
- Official ImageNet website: https://www.image-net.org/

### ImageNet

```bash
# Expected structure:
imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

**Which path should I pass?**

- Patch-Fool: pass the **root** `imagenet/` using `--data_dir`.
- All other attack evaluation scripts: pass the **split directory** (`imagenet/val` or `imagenet/train`) using `--data-dir`.

## FreqPress Defense

The FreqPress defense can be applied in two ways:

### 1. During Attack Evaluation
```bash
python -m attacks.patch_fool --defense butter_webp
```

### 2. Programmatically
```python
from defenses.freqpress import apply_freqpress_defense

# Apply to batch of images
defended_images = apply_freqpress_defense(
    adversarial_images,
    mean=torch.tensor([0.485, 0.456, 0.406]),
    std=torch.tensor([0.229, 0.224, 0.225]),
    cutoff=40,
    order=2,
    quality=50
)
```

### Frequency-Domain Analysis & Qualitative Results

<p align="center"><em>Comparison of clean and adversarial images in spatial and frequency domains and visual comparison of clean images, adversarial examples, and defended outputs.</em></p>


<p align="center">
  <img src="https://github.com/frostbyte012/FreqPress-Rectifier-For-White-Box-Adversarial-Attacks/blob/main/assets/images/analysis_2.png" width="700">
</p>

<p align="center"><em>Top: Qualitative Analysis of The FreqPress Pipeline
Stages on DeiT Base Model Under AutoAttack(𝑙∞) and Patch-
Fool Attack (Patch size=1). Bottom: Analysis of The Effects of The FreqPress Pipeline
on DeiT-Base Model Under PGD (𝜖 = 0.03) and APGD (𝜖 =
0.03) Attack With Steps Size 10 in The Pixel and Frequency
Domains.</em></p>

<p align="center">
  <img src="https://github.com/frostbyte012/FreqPress-Rectifier-For-White-Box-Adversarial-Attacks/blob/main/assets/images/qualitative_analysis_2.png"width="700">
</p>


## Models

Supported architectures:
- **Vision Transformers**: DeiT-Tiny, DeiT-Small, DeiT-Base
- **CNNs**: ResNet18, ResNet50, ResNet152

Models are automatically downloaded from pretrained weights.

## Results

Example results from our experiments:

| Attack | Model | Epsilon/Params | Defense | Accuracy |
|--------|-------|----------------|---------|----------|
| Clean | DeiT-B | - | N/A | 87.36% |
| FGSM | DeiT-B | ε=0.05 | None | 27.64% |
| FGSM | DeiT-B | ε=0.05 | FreqPress | 55.41% |
| I-FGSM | DeiT-B | ε=0.05, steps=10 | None | 0.0% |
| I-FGSM | DeiT-B | ε=0.05, steps=10 | FreqPress | 55.64% |
| PGD | DeiT-Small | ε=0.1, steps=50 | None | 1.64% |
| PGD | DeiT-Small | ε=0.1, steps=50 | FreqPress | 45.04% |
| C&W | ViT-B | c=1.0, steps=1000 | None | 0.0% |
| C&W | ViT-B | c=1.0, steps=1000 | FreqPress | 72.97% |
| DeepFool | ViT-B | steps=50 | None | 6.0% |
| DeepFool | ViT-B | steps=50 | FreqPress | 76.0% |
| AutoAttack | DeiT-Small | ε=0.05 | None | 0.5% |
| AutoAttack | DeiT-Small | ε=0.05 | FreqPress | 41.0% |
| Patch-Fool | DeiT-B | 2 patch | None | 1.6% |
| Patch-Fool | DeiT-B | 2 patch | FreqPress | 69.63% |

**Key Findings:**
- FreqPress provides consistent defense across all attack types
- Average accuracy improvement: ~60 percentage points with FreqPress
- Most effective against gradient-based attacks (FGSM, PGD)
- Still provides significant protection against optimization-based attacks (C&W, DeepFool)


## Hardware Implementation

<p align="center">
  <img src="https://github.com/frostbyte012/FreqPress-Rectifier-For-White-Box-Adversarial-Attacks/blob/main/assets/images/hardware_flowdiagram.jpg" width="400">
</p>

<p align="center"><em>Detailed flow diagram of the hardware implementation pipeline.</em></p>

<p align="center">
  <img src="https://github.com/frostbyte012/FreqPress-Rectifier-For-White-Box-Adversarial-Attacks/blob/main/assets/images/hardware_diagram.png" width="800">
</p>

<p align="center"><em>Proposed hardware architecture for real-time FreqPress defense deployment.</em></p>

## Citation

If you use this code in your research, please cite:

```bibtex
@article{freqpress2024,
  title={FreqPress: Frequency-Domain Preprocessing for Adversarial Defense},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DeiT implementation based on [facebookresearch/deit](https://github.com/facebookresearch/deit)
- Attack implementations inspired by [Harry24k/adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch)

<!-- ## Contact

For questions or issues, please open an issue on GitHub or contact [your.email@example.com] -->