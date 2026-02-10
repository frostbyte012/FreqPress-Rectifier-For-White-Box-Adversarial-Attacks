"""
Comprehensive Attack Evaluation: FGSM, I-FGSM, PGD, APGD, C&W, DeepFool, AutoAttack
Run all attacks on multiple models with and without FreqPress defense
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
from torchattacks import FGSM, BIM, PGD, APGD, CW, DeepFool, AutoAttack
import csv
from datetime import datetime

from defenses.freqpress import preprocess_batch_butterworth_webp, normalize_tensor


# Configuration defaults (can be overridden via CLI)
BATCH_SIZE = 32  # Moderate batch size for all attacks
NUM_WORKERS = 0
MAX_SAMPLES = None  # Set to None to use full dataset
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Paths (update these or override via CLI)
PATH_TO_DATASET = 'imagenet_validation'


def parse_args():
    """Parse command-line arguments for comprehensive evaluation."""
    parser = argparse.ArgumentParser(
        description="Comprehensive attack evaluation with FreqPress defense"
    )

    # Data and loader settings
    parser.add_argument("--data-dir", type=str, default=PATH_TO_DATASET,
                        help="Path to dataset root (ImageFolder structure)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="Number of DataLoader workers")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES,
                        help="Maximum number of samples to use (None = all)")

    # Model selection
    parser.add_argument("--model", type=str, default="resnet18",
                        help="Single model name to evaluate (timm identifier)")

    # FreqPress defense settings
    parser.add_argument("--cutoff", type=float, default=40.0,
                        help="Butterworth low-pass cutoff frequency")
    parser.add_argument("--order", type=int, default=2,
                        help="Butterworth filter order")
    parser.add_argument("--quality", type=int, default=70,
                        help="WebP compression quality (0-100)")

    # Control which expensive attacks to run
    parser.add_argument("--run-cw", action="store_true", default=True,
                        help="Run C&W attack")
    parser.add_argument("--no-run-cw", dest="run_cw", action="store_false",
                        help="Disable C&W attack")
    parser.add_argument("--run-deepfool", action="store_true", default=True,
                        help="Run DeepFool attack")
    parser.add_argument("--no-run-deepfool", dest="run_deepfool", action="store_false",
                        help="Disable DeepFool attack")
    parser.add_argument("--run-autoattack", action="store_true", default=False,
                        help="Run AutoAttack (very slow)")

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


def evaluate_with_defense(model, loader, attack, device, attack_name="",
                          cutoff=40.0, order=2, quality=70):
    """Evaluate model with FreqPress defense"""
    model.eval()
    correct = 0
    total = 0
    
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        
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
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  [{attack_name}] Batch {batch_idx+1}/{len(loader)}, Acc: {correct/total:.4f}")
    
    acc = correct / max(1, total)
    return acc


def evaluate_without_defense(model, loader, attack, device, attack_name=""):
    """Evaluate model without defense"""
    model.eval()
    correct = 0
    total = 0
    
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        
        output = model(x)
        init_pred = output.argmax(dim=1)
        # Generate adversarial examples
        x_adv = attack(x, init_pred)
        
        # Final prediction (no defense)
        final_pred = model(x_adv)
        preds = final_pred.argmax(dim=1)
        
        correct += (preds == y).sum().item()
        total += y.size(0)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  [{attack_name}] Batch {batch_idx+1}/{len(loader)}, Acc: {correct/total:.4f}")
    
    acc = correct / max(1, total)
    return acc


def evaluate_clean_accuracy(model, loader, device):
    """Evaluate model on clean images"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            output = model(x)
            preds = output.argmax(dim=1)
            
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    acc = correct / max(1, total)
    return acc


def run_fgsm_experiments(model, model_name, loader, device, results,
                         cutoff=40.0, order=2, quality=70):
    """Run FGSM experiments"""
    print("\n" + "="*80)
    print(f"FGSM Experiments - {model_name}")
    print("="*80)
    
    eps_values = [0.01, 0.03, 0.05]
    
    for eps in eps_values:
        attack = FGSM(model, eps=eps)
        
        # With defense
        print(f"\nFGSM eps={eps:.3f}, Defense=YES")
        acc_def = evaluate_with_defense(
            model, loader, attack, device, f"FGSM-{eps}",
            cutoff=cutoff, order=order, quality=quality
        )
        results.append({
            'model': model_name,
            'attack': 'FGSM',
            'epsilon': eps,
            'steps': 1,
            'other_params': '-',
            'defense': 'YES',
            'accuracy': acc_def
        })
        print(f"Final: FGSM eps={eps:.3f}, Defense=YES: {acc_def:.4f}")
        
        # Without defense
        print(f"\nFGSM eps={eps:.3f}, Defense=NO")
        acc_no_def = evaluate_without_defense(model, loader, attack, device, f"FGSM-{eps}")
        results.append({
            'model': model_name,
            'attack': 'FGSM',
            'epsilon': eps,
            'steps': 1,
            'other_params': '-',
            'defense': 'NO',
            'accuracy': acc_no_def
        })
        print(f"Final: FGSM eps={eps:.3f}, Defense=NO: {acc_no_def:.4f}")


def run_ifgsm_experiments(model, model_name, loader, device, results,
                          cutoff=40.0, order=2, quality=70):
    """Run I-FGSM (BIM) experiments"""
    print("\n" + "="*80)
    print(f"I-FGSM Experiments - {model_name}")
    print("="*80)
    
    configs = [
        (0.03, 5),
        (0.05, 10),
    ]
    
    for eps, steps in configs:
        alpha = eps / steps
        attack = BIM(model, eps=eps, alpha=alpha, steps=steps)
        
        # With defense
        print(f"\nI-FGSM eps={eps:.3f}, steps={steps}, Defense=YES")
        acc_def = evaluate_with_defense(
            model, loader, attack, device, f"I-FGSM-{eps}",
            cutoff=cutoff, order=order, quality=quality
        )
        results.append({
            'model': model_name,
            'attack': 'I-FGSM',
            'epsilon': eps,
            'steps': steps,
            'other_params': f'alpha={alpha:.4f}',
            'defense': 'YES',
            'accuracy': acc_def
        })
        print(f"Final: I-FGSM eps={eps:.3f}, steps={steps}, Defense=YES: {acc_def:.4f}")
        
        # Without defense
        print(f"\nI-FGSM eps={eps:.3f}, steps={steps}, Defense=NO")
        acc_no_def = evaluate_without_defense(model, loader, attack, device, f"I-FGSM-{eps}")
        results.append({
            'model': model_name,
            'attack': 'I-FGSM',
            'epsilon': eps,
            'steps': steps,
            'other_params': f'alpha={alpha:.4f}',
            'defense': 'NO',
            'accuracy': acc_no_def
        })
        print(f"Final: I-FGSM eps={eps:.3f}, steps={steps}, Defense=NO: {acc_no_def:.4f}")


def run_pgd_experiments(model, model_name, loader, device, results,
                        cutoff=40.0, order=2, quality=70):
    """Run PGD experiments"""
    print("\n" + "="*80)
    print(f"PGD Experiments - {model_name}")
    print("="*80)
    
    configs = [
        (0.03, 10),
        (0.05, 10),
    ]
    
    for eps, steps in configs:
        alpha = min(2/255, eps/steps)
        attack = PGD(model, eps=eps, alpha=alpha, steps=steps)
        
        # With defense
        print(f"\nPGD eps={eps:.3f}, steps={steps}, Defense=YES")
        acc_def = evaluate_with_defense(
            model, loader, attack, device, f"PGD-{eps}",
            cutoff=cutoff, order=order, quality=quality
        )
        results.append({
            'model': model_name,
            'attack': 'PGD',
            'epsilon': eps,
            'steps': steps,
            'other_params': f'alpha={alpha:.4f}',
            'defense': 'YES',
            'accuracy': acc_def
        })
        print(f"Final: PGD eps={eps:.3f}, steps={steps}, Defense=YES: {acc_def:.4f}")
        
        # Without defense
        print(f"\nPGD eps={eps:.3f}, steps={steps}, Defense=NO")
        acc_no_def = evaluate_without_defense(model, loader, attack, device, f"PGD-{eps}")
        results.append({
            'model': model_name,
            'attack': 'PGD',
            'epsilon': eps,
            'steps': steps,
            'other_params': f'alpha={alpha:.4f}',
            'defense': 'NO',
            'accuracy': acc_no_def
        })
        print(f"Final: PGD eps={eps:.3f}, steps={steps}, Defense=NO: {acc_no_def:.4f}")


def run_apgd_experiments(model, model_name, loader, device, results,
                         cutoff=40.0, order=2, quality=70):
    """Run APGD experiments"""
    print("\n" + "="*80)
    print(f"APGD Experiments - {model_name}")
    print("="*80)
    
    configs = [
        (0.05, 10, 2),
    ]
    
    for eps, steps, n_restarts in configs:
        attack = APGD(model, eps=eps, steps=steps, n_restarts=n_restarts)
        
        # With defense
        print(f"\nAPGD eps={eps:.3f}, steps={steps}, Defense=YES")
        acc_def = evaluate_with_defense(
            model, loader, attack, device, f"APGD-{eps}",
            cutoff=cutoff, order=order, quality=quality
        )
        results.append({
            'model': model_name,
            'attack': 'APGD',
            'epsilon': eps,
            'steps': steps,
            'other_params': f'restarts={n_restarts}',
            'defense': 'YES',
            'accuracy': acc_def
        })
        print(f"Final: APGD eps={eps:.3f}, steps={steps}, Defense=YES: {acc_def:.4f}")
        
        # Without defense
        print(f"\nAPGD eps={eps:.3f}, steps={steps}, Defense=NO")
        acc_no_def = evaluate_without_defense(model, loader, attack, device, f"APGD-{eps}")
        results.append({
            'model': model_name,
            'attack': 'APGD',
            'epsilon': eps,
            'steps': steps,
            'other_params': f'restarts={n_restarts}',
            'defense': 'NO',
            'accuracy': acc_no_def
        })
        print(f"Final: APGD eps={eps:.3f}, steps={steps}, Defense=NO: {acc_no_def:.4f}")


def run_cw_experiments(model, model_name, loader, device, results,
                       cutoff=40.0, order=2, quality=70):
    """Run Carlini-Wagner experiments"""
    print("\n" + "="*80)
    print(f"C&W Experiments - {model_name}")
    print("="*80)
    
    configs = [
        (1.0, 100),
    ]
    
    for c, steps in configs:
        print(f"\nC&W c={c}, steps={steps}")
        attack = CW(model, c=c, kappa=0, steps=steps, lr=0.01)
        
        # With defense
        print(f"C&W c={c}, steps={steps}, Defense=YES")
        acc_def = evaluate_with_defense(
            model, loader, attack, device, f"C&W-{c}",
            cutoff=cutoff, order=order, quality=quality
        )
        results.append({
            'model': model_name,
            'attack': 'C&W',
            'epsilon': '-',
            'steps': steps,
            'other_params': f'c={c}',
            'defense': 'YES',
            'accuracy': acc_def
        })
        print(f"Final: C&W c={c}, steps={steps}, Defense=YES: {acc_def:.4f}")
        
        # Without defense
        print(f"C&W c={c}, steps={steps}, Defense=NO")
        acc_no_def = evaluate_without_defense(model, loader, attack, device, f"C&W-{c}")
        results.append({
            'model': model_name,
            'attack': 'C&W',
            'epsilon': '-',
            'steps': steps,
            'other_params': f'c={c}',
            'defense': 'NO',
            'accuracy': acc_no_def
        })
        print(f"Final: C&W c={c}, steps={steps}, Defense=NO: {acc_no_def:.4f}")


def run_deepfool_experiments(model, model_name, loader, device, results,
                             cutoff=40.0, order=2, quality=70):
    """Run DeepFool experiments"""
    print("\n" + "="*80)
    print(f"DeepFool Experiments - {model_name}")
    print("="*80)
    
    configs = [
        (50, 0.02),
    ]
    
    for steps, overshoot in configs:
        print(f"\nDeepFool steps={steps}, overshoot={overshoot}")
        attack = DeepFool(model, steps=steps, overshoot=overshoot)
        
        # With defense
        print(f"DeepFool steps={steps}, overshoot={overshoot}, Defense=YES")
        acc_def = evaluate_with_defense(
            model, loader, attack, device, "DeepFool",
            cutoff=cutoff, order=order, quality=quality
        )
        results.append({
            'model': model_name,
            'attack': 'DeepFool',
            'epsilon': '-',
            'steps': steps,
            'other_params': f'overshoot={overshoot}',
            'defense': 'YES',
            'accuracy': acc_def
        })
        print(f"Final: DeepFool steps={steps}, overshoot={overshoot}, Defense=YES: {acc_def:.4f}")
        
        # Without defense
        print(f"DeepFool steps={steps}, overshoot={overshoot}, Defense=NO")
        acc_no_def = evaluate_without_defense(model, loader, attack, device, f"DeepFool")
        results.append({
            'model': model_name,
            'attack': 'DeepFool',
            'epsilon': '-',
            'steps': steps,
            'other_params': f'overshoot={overshoot}',
            'defense': 'NO',
            'accuracy': acc_no_def
        })
        print(f"Final: DeepFool steps={steps}, overshoot={overshoot}, Defense=NO: {acc_no_def:.4f}")


def run_autoattack_experiments(model, model_name, loader, device, results,
                               cutoff=40.0, order=2, quality=70):
    """Run AutoAttack experiments"""
    print("\n" + "="*80)
    print(f"AutoAttack Experiments - {model_name}")
    print("="*80)
    print("⚠️  WARNING: AutoAttack is very slow!")
    
    configs = [
        (0.03, 'standard'),
    ]
    
    for eps, version in configs:
        print(f"\nAutoAttack eps={eps}, version={version}")
        attack = AutoAttack(model, norm='Linf', eps=eps, version=version, verbose=False)
        
        # With defense
        print(f"AutoAttack eps={eps}, version={version}, Defense=YES")
        acc_def = evaluate_with_defense(
            model, loader, attack, device, f"AutoAttack-{eps}",
            cutoff=cutoff, order=order, quality=quality
        )
        results.append({
            'model': model_name,
            'attack': 'AutoAttack',
            'epsilon': eps,
            'steps': '-',
            'other_params': f'version={version}',
            'defense': 'YES',
            'accuracy': acc_def
        })
        print(f"Final: AutoAttack eps={eps}, version={version}, Defense=YES: {acc_def:.4f}")
        
        # Without defense
        print(f"AutoAttack eps={eps}, version={version}, Defense=NO")
        acc_no_def = evaluate_without_defense(model, loader, attack, device, f"AutoAttack-{eps}")
        results.append({
            'model': model_name,
            'attack': 'AutoAttack',
            'epsilon': eps,
            'steps': '-',
            'other_params': f'version={version}',
            'defense': 'NO',
            'accuracy': acc_no_def
        })
        print(f"Final: AutoAttack eps={eps}, version={version}, Defense=NO: {acc_no_def:.4f}")


def save_results_to_csv(results, output_path):
    """Save results to CSV file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['model', 'attack', 'epsilon', 'steps', 'other_params', 'defense', 'accuracy']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {output_path}")


def main():
    args = parse_args()

    # Setup
    setup_reproducibility()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    print(f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    
    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    ds = load_dataset(args.data_dir, args.max_samples)
    loader = DataLoader(
        ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=(device.type == 'cuda')
    )
    print(f"Dataset loaded: {len(ds)} samples, {len(loader)} batches\n")
    
    # Model to evaluate (single model from CLI)
    models_list = [args.model]

    results = []
    
    # Evaluate each model
    for model_name in models_list:
        print("\n" + "#"*80)
        print(f"# EVALUATING MODEL: {model_name}")
        print("#"*80)
        
        # Load model
        model = timm.create_model(model_name, pretrained=True)
        model = model.to(device)
        model.eval()
        
        # Clean accuracy
        print("\nEvaluating clean accuracy...")
        clean_acc = evaluate_clean_accuracy(model, loader, device)
        print(f"Clean Accuracy: {clean_acc:.4f}")
        results.append({
            'model': model_name,
            'attack': 'Clean',
            'epsilon': 0,
            'steps': 0,
            'other_params': '-',
            'defense': 'N/A',
            'accuracy': clean_acc
        })
        
        # Run all attacks
        run_fgsm_experiments(
            model, model_name, loader, device, results,
            cutoff=args.cutoff, order=args.order, quality=args.quality
        )
        run_ifgsm_experiments(
            model, model_name, loader, device, results,
            cutoff=args.cutoff, order=args.order, quality=args.quality
        )
        run_pgd_experiments(
            model, model_name, loader, device, results,
            cutoff=args.cutoff, order=args.order, quality=args.quality
        )
        run_apgd_experiments(
            model, model_name, loader, device, results,
            cutoff=args.cutoff, order=args.order, quality=args.quality
        )
        
        if args.run_cw:
            run_cw_experiments(
                model, model_name, loader, device, results,
                cutoff=args.cutoff, order=args.order, quality=args.quality
            )
        else:
            print("\n⏭️  Skipping C&W (RUN_CW=False)")
            
        if args.run_deepfool:
            run_deepfool_experiments(
                model, model_name, loader, device, results,
                cutoff=args.cutoff, order=args.order, quality=args.quality
            )
        else:
            print("\n⏭️  Skipping DeepFool (RUN_DEEPFOOL=False)")
            
        if args.run_autoattack:
            run_autoattack_experiments(
                model, model_name, loader, device, results,
                cutoff=args.cutoff, order=args.order, quality=args.quality
            )
        else:
            print("\n⏭️  Skipping AutoAttack (RUN_AUTOATTACK=False)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/comprehensive_evaluation_{timestamp}.csv"
    save_results_to_csv(results, output_path)
    
    # Print summary table
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*80)
    print(f"{'Model':<25} {'Attack':<12} {'Eps':<8} {'Steps':<6} {'Defense':<8} {'Acc':<8}")
    print("-"*80)
    for r in results:
        eps_str = str(r['epsilon']) if isinstance(r['epsilon'], (int, float)) else r['epsilon']
        steps_str = str(r['steps']) if isinstance(r['steps'], (int, float)) else r['steps']
        print(f"{r['model']:<25} {r['attack']:<12} {eps_str:<8} {steps_str:<6} "
              f"{r['defense']:<8} {r['accuracy']:<8.4f}")


if __name__ == "__main__":
    # Optional: Setup logging to file
    # setup_logging("/path/to/comprehensive_evaluation.log")
    main()