"""
Patch-Fool: Attention-based patch adversarial attack
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import numpy as np
import time
import csv
from torchvision import transforms
from PIL import Image

from models.DeiT import deit_base_patch16_224, deit_tiny_patch16_224, deit_small_patch16_224
from models.resnet import ResNet50, ResNet152, ResNet101
from utils.data_loader import get_loaders
from utils.logger import my_logger
from utils.meter import my_meter
from utils.ops import clamp, PCGrad
from defenses.freqpress import apply_freqpress_defense, save_unnormalized_tensor_image


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Patch-Fool Training')

    # Basic settings
    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--dataset', default='ImageNet', type=str)
    parser.add_argument('--data_dir', default='/data1/ImageNet/ILSVRC/Data/CLS-LOC/', type=str)
    parser.add_argument('--log_dir', default='log', type=str)
    parser.add_argument('--crop_size', default=224, type=int)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--workers', default=16, type=int)
    parser.add_argument('--defense', default='none', type=str, choices=['none', 'butter_webp'],
                        help='Apply post-attack image transformation before evaluation')

    # Model settings
    parser.add_argument('--network', default='DeiT-B', type=str, 
                        choices=['DeiT-B', 'DeiT-S', 'DeiT-T', 'ResNet152', 'ResNet50', 'ResNet18'])
    parser.add_argument('--dataset_size', default=1.0, type=float, help='Use part of Eval set')

    # Patch selection
    parser.add_argument('--patch_select', default='Attn', type=str, choices=['Rand', 'Saliency', 'Attn'])
    parser.add_argument('--num_patch', default=1, type=int)
    parser.add_argument('--sparse_pixel_num', default=0, type=int)

    # Attack settings
    parser.add_argument('--attack_mode', default='CE_loss', choices=['CE_loss', 'Attention'], type=str)
    parser.add_argument('--atten_loss_weight', default=0.002, type=float)
    parser.add_argument('--atten_select', default=4, type=int, help='Select patch based on which attention layer')
    parser.add_argument('--mild_l_2', default=0., type=float, help='Range: 0-16')
    parser.add_argument('--mild_l_inf', default=0., type=float, help='Range: 0-1')

    # Training settings
    parser.add_argument('--train_attack_iters', default=250, type=int)
    parser.add_argument('--random_sparse_pixel', action='store_true', help='random select sparse pixel or not')
    parser.add_argument('--learnable_mask_stop', default=200, type=int)
    parser.add_argument('--attack_learning_rate', default=0.22, type=float)
    parser.add_argument('--step_size', default=10, type=int)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    # Single-image evaluation mode
    parser.add_argument('--single_image', action='store_true',
                        help='Run Patch-Fool attack on a single image instead of the dataset')
    parser.add_argument('--single_image_path', default='', type=str,
                        help='Path to the input image when using --single_image')
    parser.add_argument('--single_image_label', default=None, type=int,
                        help='Optional ground-truth ImageNet class index for the single image. '
                             'If not set, the model prediction is used as label.')

    args = parser.parse_args()

    # Validate constraint arguments
    if args.mild_l_2 != 0 and args.mild_l_inf != 0:
        print(f'Only one parameter can be non-zero: mild_l_2 {args.mild_l_2}, mild_l_inf {args.mild_l_inf}')
        raise NotImplementedError
    if args.mild_l_inf > 1:
        args.mild_l_inf /= 255.
        print(f'mild_l_inf > 1. Constrain all the perturbation with mild_l_inf/255={args.mild_l_inf}')

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    return args


def forward_logits(model, x, args):
    """Forward helper that always returns logits tensor (handles DeiT vs ResNet)"""
    if 'DeiT' in args.network:
        out, _ = model(x)
        return out
    else:
        return model(x)


def main():
    args = get_args()
    logger = my_logger(args)
    meter = my_meter()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    patch_size = 16    
    filter = torch.ones([1, 3, patch_size, patch_size]).float().cuda()

    # Load model
    if args.network == 'ResNet152':
        model = ResNet152(pretrained=True)
    elif args.network == 'ResNet50':
        model = ResNet50(pretrained=True)
    elif args.network == 'ResNet18':
        model = torchvision.models.resnet18(pretrained=True)  
    elif args.network == 'DeiT-T':
        model = deit_tiny_patch16_224(pretrained=True)
    elif args.network == 'DeiT-S':
        model = deit_small_patch16_224(pretrained=True)
    elif args.network == 'DeiT-B':
        model = deit_base_patch16_224(pretrained=True)
    else:
        print('Wrong Network')
        raise ValueError(f"Unknown network: {args.network}")

    model = model.cuda()
    model = torch.nn.DataParallel(model)
    model.eval()

    criterion = nn.CrossEntropyLoss().cuda()
    
    # Setup data loader
    if args.single_image:
        if args.single_image_path == '':
            raise ValueError('When --single_image is set, --single_image_path must be provided.')

        # Build same preprocessing as in utils.get_loaders
        loader = get_loaders(args)  # Initialize to set args.mu and args.std
        
        preprocess = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mu, std=args.std),
        ])

        img = Image.open(args.single_image_path).convert('RGB')
        X = preprocess(img).unsqueeze(0).cuda()

        # Determine label
        with torch.no_grad():
            logits = forward_logits(model, X, args)
            pred_label = logits.argmax(dim=1)
        if args.single_image_label is not None:
            y = torch.tensor([args.single_image_label], dtype=torch.long).cuda()
        else:
            y = pred_label

        loader = [(X, y)]
    else:
        loader = get_loaders(args)
    
    mu = torch.tensor(args.mu).view(3, 1, 1).cuda()
    std = torch.tensor(args.std).view(3, 1, 1).cuda()

    start_time = time.time()
    false2true_num = 0

    # Setup single-image output directory
    if args.single_image:
        single_out_dir = os.path.join(args.log_dir, "single_image_results")
        os.makedirs(single_out_dir, exist_ok=True)
        single_results = []

    for i, (X, y) in enumerate(loader):
        if i == int(len(loader) * args.dataset_size):
            break

        X, y = X.cuda(), y.cuda()
        patch_num_per_line = int(X.size(-1) / patch_size)
        delta = torch.zeros_like(X).cuda()
        delta.requires_grad = True

        model.zero_grad()
        if 'DeiT' in args.network:
            out, atten = model(X + delta)
        else:
            out = model(X + delta)

        classification_result = out.max(1)[1] == y
        correct_num = classification_result.sum().item()
        loss = criterion(out, y)
        meter.add_loss_acc("Base", {'CE': loss.item()}, correct_num, y.size(0))

        # Choose patch
        if args.patch_select == 'Rand':
            max_patch_index = np.random.randint(0, 14 * 14, (X.size(0), args.num_patch))
            max_patch_index = torch.from_numpy(max_patch_index)
        elif args.patch_select == 'Saliency':
            grad = torch.autograd.grad(loss, delta)[0]
            grad = torch.abs(grad)
            patch_grad = F.conv2d(grad, filter, stride=patch_size)
            patch_grad = patch_grad.view(patch_grad.size(0), -1)
            max_patch_index = patch_grad.argsort(descending=True)[:, :args.num_patch]
        elif args.patch_select == 'Attn':
            atten_layer = atten[args.atten_select].mean(dim=1)
            if 'DeiT' in args.network:
                atten_layer = atten_layer.mean(dim=-2)[:, 1:]
            else:
                atten_layer = atten_layer.mean(dim=-2)
            max_patch_index = atten_layer.argsort(descending=True)[:, :args.num_patch]
        else:
            print(f'Unknown patch_select: {args.patch_select}')
            raise ValueError(f"Unknown patch_select: {args.patch_select}")

        # Build mask
        mask = torch.zeros([X.size(0), 1, X.size(2), X.size(3)]).cuda()
        if args.sparse_pixel_num != 0:
            learnable_mask = mask.clone()

        for j in range(X.size(0)):
            index_list = max_patch_index[j]
            for index in index_list:
                row = (index // patch_num_per_line) * patch_size
                column = (index % patch_num_per_line) * patch_size

                if args.sparse_pixel_num != 0:
                    learnable_mask.data[j, :, row:row + patch_size, column:column + patch_size] = torch.rand(
                        [patch_size, patch_size])
                mask[j, :, row:row + patch_size, column:column + patch_size] = 1

        # Initialize adversarial attack
        max_patch_index_matrix = max_patch_index[:, 0]
        max_patch_index_matrix = max_patch_index_matrix.repeat(197, 1)
        max_patch_index_matrix = max_patch_index_matrix.permute(1, 0)
        max_patch_index_matrix = max_patch_index_matrix.flatten().long()

        if args.mild_l_inf == 0:
            delta = (torch.rand_like(X) - mu) / std
        else:
            epsilon = args.mild_l_inf / std
            delta = 2 * epsilon * torch.rand_like(X) - epsilon + X

        delta.data = clamp(delta, (0 - mu) / std, (1 - mu) / std)
        original_img = X.clone()

        if args.random_sparse_pixel:
            sparse_mask = torch.zeros_like(mask)
            learnable_mask_temp = learnable_mask.view(learnable_mask.size(0), -1)
            sparse_mask_temp = sparse_mask.view(sparse_mask.size(0), -1)
            value, _ = learnable_mask_temp.sort(descending=True)
            threshold = value[:, args.sparse_pixel_num - 1].view(-1, 1)
            sparse_mask_temp[learnable_mask_temp >= threshold] = 1
            mask = sparse_mask

        if args.sparse_pixel_num == 0 or args.random_sparse_pixel:
            X = torch.mul(X, 1 - mask)
        else:
            learnable_mask.requires_grad = True
            
        delta = delta.cuda()
        delta.requires_grad = True

        opt = torch.optim.Adam([delta], lr=args.attack_learning_rate)
        if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel):
            mask_opt = torch.optim.Adam([learnable_mask], lr=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)

        # Start adversarial attack
        for train_iter_num in range(args.train_attack_iters):
            model.zero_grad()
            opt.zero_grad()

            # Build sparse patch attack binary mask
            if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel):
                if train_iter_num < args.learnable_mask_stop:
                    mask_opt.zero_grad()
                    sparse_mask = torch.zeros_like(mask)
                    learnable_mask_temp = learnable_mask.view(learnable_mask.size(0), -1)
                    sparse_mask_temp = sparse_mask.view(sparse_mask.size(0), -1)
                    value, _ = learnable_mask_temp.sort(descending=True)
                    threshold = value[:, args.sparse_pixel_num-1].view(-1, 1)
                    sparse_mask_temp[learnable_mask_temp >= threshold] = 1
                    temp_mask = ((sparse_mask - learnable_mask).detach() + learnable_mask) * mask
                else:
                    temp_mask = sparse_mask

                X = original_img * (1-sparse_mask)
                if 'DeiT' in args.network:
                    out, atten = model(X + torch.mul(delta, temp_mask))
                else:
                    out = model(X + torch.mul(delta, temp_mask))
            else:
                if 'DeiT' in args.network:
                    out, atten = model(X + torch.mul(delta, mask))
                else:
                    out = model(X + torch.mul(delta, mask))

            loss = criterion(out, y)

            if args.attack_mode == 'Attention':
                grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
                ce_loss_grad_temp = grad.view(X.size(0), -1).detach().clone()
                if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel) and train_iter_num < args.learnable_mask_stop:
                    mask_grad = torch.autograd.grad(loss, learnable_mask, retain_graph=True)[0]

                range_list = range(len(atten)//2)
                for atten_num in range_list:
                    if atten_num == 0:
                        continue
                    atten_map = atten[atten_num]
                    atten_map = atten_map.mean(dim=1)
                    atten_map = atten_map.view(-1, atten_map.size(-1))
                    atten_map = -torch.log(atten_map)
                    if 'DeiT' in args.network:
                        atten_loss = F.nll_loss(atten_map, max_patch_index_matrix + 1)
                    else:
                        atten_loss = F.nll_loss(atten_map, max_patch_index_matrix)

                    atten_grad = torch.autograd.grad(atten_loss, delta, retain_graph=True)[0]
                    atten_grad_temp = atten_grad.view(X.size(0), -1)
                    cos_sim = F.cosine_similarity(atten_grad_temp, ce_loss_grad_temp, dim=1)

                    if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel) and train_iter_num < args.learnable_mask_stop:
                        mask_atten_grad = torch.autograd.grad(atten_loss, learnable_mask, retain_graph=True)[0]

                    atten_grad = PCGrad(atten_grad_temp, ce_loss_grad_temp, cos_sim, grad.shape)
                    if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel):
                        mask_atten_grad_temp = mask_atten_grad.view(mask_atten_grad.size(0), -1)
                        ce_mask_grad_temp = mask_grad.view(mask_grad.size(0), -1)
                        mask_cos_sim = F.cosine_similarity(mask_atten_grad_temp, ce_mask_grad_temp, dim=1)
                        mask_atten_grad = PCGrad(mask_atten_grad_temp, ce_mask_grad_temp, mask_cos_sim, mask_atten_grad.shape)

                    grad += atten_grad * args.atten_loss_weight
                    if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel):
                        mask_grad += mask_atten_grad * args.atten_loss_weight
            else:
                if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel) and train_iter_num < args.learnable_mask_stop:
                    grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
                    mask_grad = torch.autograd.grad(loss, learnable_mask)[0]
                else:
                    grad = torch.autograd.grad(loss, delta)[0]

            opt.zero_grad()
            delta.grad = -grad
            opt.step()
            scheduler.step()

            if args.sparse_pixel_num != 0 and (not args.random_sparse_pixel) and train_iter_num < args.learnable_mask_stop:
                mask_opt.zero_grad()
                learnable_mask.grad = -mask_grad
                mask_opt.step()
                learnable_mask_temp = learnable_mask.view(X.size(0), -1)
                learnable_mask.data -= learnable_mask_temp.min(-1)[0].view(-1, 1, 1, 1)
                learnable_mask.data += 1e-6
                learnable_mask.data *= mask

            # L2 constraint
            if args.mild_l_2 != 0:
                radius = (args.mild_l_2 / std).squeeze()
                perturbation = (delta.detach() - original_img) * mask
                l2 = torch.linalg.norm(perturbation.view(perturbation.size(0), perturbation.size(1), -1), dim=-1)
                radius = radius.repeat([l2.size(0), 1])
                l2_constraint = radius / l2
                l2_constraint[l2 < radius] = 1.
                l2_constraint = l2_constraint.view(l2_constraint.size(0), l2_constraint.size(1), 1, 1)
                delta.data = original_img + perturbation * l2_constraint

            # L_inf constraint
            if args.mild_l_inf != 0:
                epsilon = args.mild_l_inf / std
                delta.data = clamp(delta, original_img - epsilon, original_img + epsilon)

            delta.data = clamp(delta, (0 - mu) / std, (1 - mu) / std)

        # Evaluate adversarial attack
        with torch.no_grad():
            if args.sparse_pixel_num == 0 or args.random_sparse_pixel:
                perturb_x = X + torch.mul(delta, mask)
                if 'DeiT' in args.network:
                    out, atten = model(perturb_x)
                else:
                    out = model(perturb_x)  
            else:
                if train_iter_num < args.learnable_mask_stop:
                    sparse_mask = torch.zeros_like(mask)
                    learnable_mask_temp = learnable_mask.view(learnable_mask.size(0), -1)
                    temp_mask = sparse_mask.view(sparse_mask.size(0), -1)
                    value, _ = learnable_mask_temp.sort(descending=True)
                    threshold = value[:, args.sparse_pixel_num - 1].view(-1, 1)
                    temp_mask[learnable_mask_temp >= threshold] = 1

                print((sparse_mask * mask).view(mask.size(0), -1).sum(-1))
                print("xxxxxxxxxxxxxxxxxxxxxx")
                X = original_img * (1 - sparse_mask)
                perturb_x = X + torch.mul(delta, sparse_mask)
                if 'DeiT' in args.network:
                    out, atten = model(perturb_x)
                else:
                    out = model(perturb_x)

            raw_perturb_x = perturb_x.clone().detach()

            # Apply optional defense
            if args.defense == 'butter_webp':
                defended = []
                mean = mu.view(3, 1, 1)
                stdv = std.view(3, 1, 1)
                for b in range(perturb_x.size(0)):
                    defended.append(apply_freqpress_defense(perturb_x[b], mean, stdv))
                perturb_x = torch.stack(defended)
                if 'DeiT' in args.network:
                    out, atten = model(perturb_x)
                else:
                    out = model(perturb_x)

            classification_result_after_attack = out.max(1)[1] == y
            loss = criterion(out, y)
            meter.add_loss_acc("ADV", {'CE': loss.item()}, 
                             (classification_result_after_attack.sum().item()), y.size(0))

            # Single-image: save images + predictions
            if args.single_image:
                b = 0
                mean = mu.view(3, 1, 1)
                stdv = std.view(3, 1, 1)

                def get_pred_conf(t):
                    logits = forward_logits(model, t.unsqueeze(0), args)
                    probs = F.softmax(logits, dim=1)
                    conf, pred = probs.max(dim=1)
                    return int(pred.item()), float(conf.item())

                from defenses.freqpress import (butterworth_low_pass_filter, 
                                                webp_compression)
                
                # Clean image
                clean_tensor = original_img[b].detach()
                clean_path = os.path.join(single_out_dir, "clean.png")
                save_unnormalized_tensor_image(clean_tensor, mean, stdv, clean_path)
                clean_idx, clean_conf = get_pred_conf(clean_tensor)
                single_results.append({
                    "variant": "clean",
                    "pred_idx": clean_idx,
                    "confidence": clean_conf,
                    "image_path": clean_path,
                })

                # Adversarial image
                adv_tensor = raw_perturb_x[b].detach()
                adv_path = os.path.join(single_out_dir, "adv.png")
                save_unnormalized_tensor_image(adv_tensor, mean, stdv, adv_path)
                adv_idx, adv_conf = get_pred_conf(adv_tensor)
                single_results.append({
                    "variant": "adv",
                    "pred_idx": adv_idx,
                    "confidence": adv_conf,
                    "image_path": adv_path,
                })

                # Build defended variants
                adv_img_for_pil = adv_tensor.detach().cpu()
                adv_img_for_pil = adv_img_for_pil * stdv.detach().cpu() + mean.detach().cpu()
                adv_img_for_pil = adv_img_for_pil.clamp(0.0, 1.0)
                adv_pil = transforms.ToPILImage()(adv_img_for_pil)

                clean_img_for_pil = clean_tensor.detach().cpu()
                clean_img_for_pil = clean_img_for_pil * stdv.detach().cpu() + mean.detach().cpu()
                clean_img_for_pil = clean_img_for_pil.clamp(0.0, 1.0)
                clean_pil = transforms.ToPILImage()(clean_img_for_pil)

                # Butterworth only (adv)
                bw_pil = butterworth_low_pass_filter(adv_pil, cutoff=40, order=2)
                bw_tensor = transforms.ToTensor()(bw_pil)
                bw_tensor = (bw_tensor - mean.detach().cpu()) / stdv.detach().cpu()
                bw_tensor = bw_tensor.to(adv_tensor.device)
                bw_path = os.path.join(single_out_dir, "adv_butterworth.png")
                save_unnormalized_tensor_image(bw_tensor, mean, stdv, bw_path)
                bw_idx, bw_conf = get_pred_conf(bw_tensor)
                single_results.append({
                    "variant": "adv_butterworth",
                    "pred_idx": bw_idx,
                    "confidence": bw_conf,
                    "image_path": bw_path,
                })

                # Butterworth only (clean)
                clean_bw_pil = butterworth_low_pass_filter(clean_pil, cutoff=40, order=2)
                clean_bw_tensor = transforms.ToTensor()(clean_bw_pil)
                clean_bw_tensor = (clean_bw_tensor - mean.detach().cpu()) / stdv.detach().cpu()
                clean_bw_tensor = clean_bw_tensor.to(clean_tensor.device)
                clean_bw_path = os.path.join(single_out_dir, "clean_butterworth.png")
                save_unnormalized_tensor_image(clean_bw_tensor, mean, stdv, clean_bw_path)
                clean_bw_idx, clean_bw_conf = get_pred_conf(clean_bw_tensor)
                single_results.append({
                    "variant": "clean_butterworth",
                    "pred_idx": clean_bw_idx,
                    "confidence": clean_bw_conf,
                    "image_path": clean_bw_path,
                })

                # WebP only (adv)
                webp_pil = webp_compression(adv_pil, quality=50)
                webp_tensor = transforms.ToTensor()(webp_pil)
                webp_tensor = (webp_tensor - mean.detach().cpu()) / stdv.detach().cpu()
                webp_tensor = webp_tensor.to(adv_tensor.device)
                webp_path = os.path.join(single_out_dir, "adv_webp.png")
                save_unnormalized_tensor_image(webp_tensor, mean, stdv, webp_path)
                webp_idx, webp_conf = get_pred_conf(webp_tensor)
                single_results.append({
                    "variant": "adv_webp",
                    "pred_idx": webp_idx,
                    "confidence": webp_conf,
                    "image_path": webp_path,
                })

                # WebP only (clean)
                clean_webp_pil = webp_compression(clean_pil, quality=50)
                clean_webp_tensor = transforms.ToTensor()(clean_webp_pil)
                clean_webp_tensor = (clean_webp_tensor - mean.detach().cpu()) / stdv.detach().cpu()
                clean_webp_tensor = clean_webp_tensor.to(clean_tensor.device)
                clean_webp_path = os.path.join(single_out_dir, "clean_webp.png")
                save_unnormalized_tensor_image(clean_webp_tensor, mean, stdv, clean_webp_path)
                clean_webp_idx, clean_webp_conf = get_pred_conf(clean_webp_tensor)
                single_results.append({
                    "variant": "clean_webp",
                    "pred_idx": clean_webp_idx,
                    "confidence": clean_webp_conf,
                    "image_path": clean_webp_path,
                })

                # Butterworth + WebP (adv)
                bw_webp_pil = webp_compression(bw_pil, quality=50)
                bw_webp_tensor = transforms.ToTensor()(bw_webp_pil)
                bw_webp_tensor = (bw_webp_tensor - mean.detach().cpu()) / stdv.detach().cpu()
                bw_webp_tensor = bw_webp_tensor.to(adv_tensor.device)
                bw_webp_path = os.path.join(single_out_dir, "adv_butterworth_webp.png")
                save_unnormalized_tensor_image(bw_webp_tensor, mean, stdv, bw_webp_path)
                bw_webp_idx, bw_webp_conf = get_pred_conf(bw_webp_tensor)
                single_results.append({
                    "variant": "adv_butterworth_webp",
                    "pred_idx": bw_webp_idx,
                    "confidence": bw_webp_conf,
                    "image_path": bw_webp_path,
                })

                # Butterworth + WebP (clean)
                clean_bw_webp_pil = webp_compression(clean_bw_pil, quality=50)
                clean_bw_webp_tensor = transforms.ToTensor()(clean_bw_webp_pil)
                clean_bw_webp_tensor = (clean_bw_webp_tensor - mean.detach().cpu()) / stdv.detach().cpu()
                clean_bw_webp_tensor = clean_bw_webp_tensor.to(clean_tensor.device)
                clean_bw_webp_path = os.path.join(single_out_dir, "clean_butterworth_webp.png")
                save_unnormalized_tensor_image(clean_bw_webp_tensor, mean, stdv, clean_bw_webp_path)
                clean_bw_webp_idx, clean_bw_webp_conf = get_pred_conf(clean_bw_webp_tensor)
                single_results.append({
                    "variant": "clean_butterworth_webp",
                    "pred_idx": clean_bw_webp_idx,
                    "confidence": clean_bw_webp_conf,
                    "image_path": clean_bw_webp_path,
                })

        # Message
        if i % 1 == 0:
            logger.info("Iter: [{:d}/{:d}] Loss and Acc for all models:".format(
                i, int(len(loader) * args.dataset_size)))
            msg = meter.get_loss_acc_msg()
            logger.info(msg)

            classification_result_after_attack = classification_result_after_attack[
                classification_result == False]
            false2true_num += classification_result_after_attack.sum().item()
            logger.info("Total False -> True: {}".format(false2true_num))

    # Save single-image results
    if args.single_image:
        csv_path = os.path.join(single_out_dir, "results.csv")
        with open(csv_path, "w", newline="") as f:
            fieldnames = ["variant", "pred_idx", "confidence", "image_path"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(single_results)
        logger.info(f"Single-image results saved to {csv_path}")

    end_time = time.time()
    msg = meter.get_loss_acc_msg()
    logger.info("\nFinish! Using time: {}\n{}".format((end_time - start_time), msg))


if __name__ == "__main__":
    main()
                
