"""
PGD and APGD Attack Evaluation with FreqPress Defense
"""
import os
import sys
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm
from torchattacks import APGD, PGD

from defenses.freqpress import preprocess_batch_butterworth_webp, normalize_tensor


# Configuration defaults (can be overridden via CLI)
BATCH_SIZE = 64
NUM_WORKERS = 0
MAX_SAMPLES = None  # Set to None to use full dataset
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Paths (update these or override via CLI)
PATH_TO_DATASET = 'imagenet_validation'

def parse_args():
    """Parse command-line arguments for PGD/APGD evaluation."""
    parser = argparse.ArgumentParser(
        description="PGD and APGD evaluation with FreqPress defense"
    )

    # Data and loader settings
    parser.add_argument("--data-dir", type=str, default=PATH_TO_DATASET,
                        help="Path to training/validation dataset root")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="Number of DataLoader workers")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES,
                        help="Maximum number of samples to use (None = all)")

    # Model and attack settings
    parser.add_argument("--model", type=str, default="resnet18",
                        help="Model name (timm model identifier)")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="Epsilon for PGD/APGD attacks")

    parser.add_argument("--pgd-steps", type=int, default=10,
                        help="Number of PGD steps")
    parser.add_argument("--apgd-steps", type=int, default=10,
                        help="Number of APGD steps")
    parser.add_argument("--apgd-restarts", type=int, default=2,
                        help="Number of APGD restarts")

    # FreqPress defense settings
    parser.add_argument("--cutoff", type=float, default=40.0,
                        help="Butterworth low-pass cutoff frequency")
    parser.add_argument("--order", type=int, default=2,
                        help="Butterworth filter order")
    parser.add_argument("--quality", type=int, default=70,
                        help="WebP compression quality (0-100)")

    return parser.parse_args()


def setup_logging(log_path):
    """Setup logging to file"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    sys.stdout = open(log_path, "w", buffering=1)
    sys.stderr = sys.stdout
    print("Logging started...\n")


def setup_reproducibility(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_dataset(path, max_samples=None):
    """Load ImageNet validation dataset"""
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    ds = ImageFolder(path, transform=base_transform)
    
    if max_samples is not None:
        idx = list(range(min(max_samples, len(ds))))
        ds = Subset(ds, idx)
    
    return ds


def evaluate_pipeline(model, loader, attack, device, cutoff=40.0, order=2, quality=70):
    """
    Evaluate model with FreqPress defense against adversarial attack
    
    Args:
        model: Neural network model
        loader: Data loader
        attack: Attack method (PGD or APGD)
        device: Device to run on
    
    Returns:
        Accuracy after defense
    """
    model.eval()
    correct = 0
    total = 0
    
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        
        # Get initial prediction
        output = model(x)
        init_pred = output.argmax(dim=1)
        
        # Generate adversarial examples
        x_adv = attack(x, init_pred)
        
        # Apply FreqPress defense
        processed = preprocess_batch_butterworth_webp(
            x_adv, cutoff=cutoff, order=order, quality=quality
        )
        
        # Final prediction
        final_pred = model(processed)
        preds = final_pred.argmax(dim=1)
        
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    acc = correct / max(1, total)
    return acc


def evaluate_pipeline_no_defense(model, loader, attack, device):
    """
    Evaluate model without defense against adversarial attack
    
    Args:
        model: Neural network model
        loader: Data loader
        attack: Attack method (PGD or APGD)
        device: Device to run on
    
    Returns:
        Accuracy without defense
    """
    model.eval()
    correct = 0
    total = 0
    
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        
        # Get initial prediction
        output = model(x)
        init_pred = output.argmax(dim=1)
        
        # Generate adversarial examples
        x_adv = attack(x, init_pred)
        
        # Final prediction (no defense)
        final_pred = model(x_adv)
        preds = final_pred.argmax(dim=1)
        
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    acc = correct / max(1, total)
    return acc


def main():
    args = parse_args()

    # Setup
    setup_reproducibility()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    
    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    ds = load_dataset(args.data_dir, args.max_samples)
    loader = DataLoader(
        ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=(device.type == 'cuda')
    )
    print(f"Dataset loaded: {len(ds)} samples, {len(loader)} batches")
    
    # Experiment configuration from CLI
    eps = args.epsilon
    pgd_steps = args.pgd_steps
    apgd_steps = args.apgd_steps
    apgd_restarts = args.apgd_restarts
    model_name = args.model
    
    results = []
    
    # Load model once (single configuration)
    print("\n" + "="*80)
    print("APGD and PGD Attacks")
    print("="*80)

    print(f"\nLoading model: {model_name}")
    model = timm.create_model(model_name, pretrained=True)
    model = model.to(device)
    model.eval()

    # APGD Attack with FreqPress
    print("\n" + "="*80)
    print("APGD Attack - with FreqPress Defense")
    print("="*80)

    attack_apgd = APGD(model, eps=eps, steps=apgd_steps, n_restarts=apgd_restarts)
    acc = evaluate_pipeline(
        model, loader, attack_apgd, device,
        cutoff=args.cutoff, order=args.order, quality=args.quality
    )
    result_str = (
        f'Model: {model_name}, Epsilon: {eps}, Steps: {apgd_steps}, '
        f'Restarts: {apgd_restarts}, Acc: {acc:.4f}'
    )
    print(result_str)
    results.append(result_str)

    # PGD Attack with FreqPress
    print("\n" + "="*80)
    print("PGD Attack - with FreqPress Defense")
    print("="*80)

    alpha = min(2/255, eps/pgd_steps) if pgd_steps > 0 else 2/255
    attack_pgd = PGD(model, eps=eps, steps=pgd_steps, alpha=alpha)
    acc = evaluate_pipeline(
        model, loader, attack_pgd, device,
        cutoff=args.cutoff, order=args.order, quality=args.quality
    )
    result_str = f'Model: {model_name}, Epsilon: {eps}, Steps: {pgd_steps}, Acc: {acc:.4f}'
    print(result_str)
    results.append(result_str)

    # APGD Attack without Defense
    print("\n" + "="*80)
    print("APGD Attack - NO Defense")
    print("="*80)

    attack_apgd = APGD(model, eps=eps, steps=apgd_steps, n_restarts=apgd_restarts)
    acc = evaluate_pipeline_no_defense(model, loader, attack_apgd, device)
    result_str = (
        f'Model: {model_name}, Epsilon: {eps}, Steps: {apgd_steps}, '
        f'Restarts: {apgd_restarts}, Acc: {acc:.4f}'
    )
    print(result_str)
    results.append(result_str)

    # PGD Attack without Defense
    print("\n" + "="*80)
    print("PGD Attack - NO Defense")
    print("="*80)

    attack_pgd = PGD(model, eps=eps, steps=pgd_steps, alpha=alpha)
    acc = evaluate_pipeline_no_defense(model, loader, attack_pgd, device)
    result_str = f'Model: {model_name}, Epsilon: {eps}, Steps: {pgd_steps}, Acc: {acc:.4f}'
    print(result_str)
    results.append(result_str)
    
    # Print summary
    print("\n" + "="*80)
    print("ALL RESULTS SUMMARY")
    print("="*80)
    for result in results:
        print(result)


if __name__ == "__main__":
    # Optional: Setup logging to file
    # setup_logging("/path/to/pgd_evaluation.log")
    main()
