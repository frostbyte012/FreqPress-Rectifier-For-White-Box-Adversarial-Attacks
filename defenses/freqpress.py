"""
FreqPress Defense: Butterworth Low-Pass Filter + WebP Compression
"""
import io
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def webp_compression(img, quality=50):
    """
    Apply WebP compression to image
    
    Args:
        img: PIL Image or torch.Tensor
        quality: WebP compression quality (0-100)
    
    Returns:
        PIL Image after WebP compression
    """
    if isinstance(img, torch.Tensor):
        img = transforms.ToPILImage()(img.cpu().squeeze().clamp(0, 1))
    elif not isinstance(img, Image.Image):
        raise TypeError('Input should be a PIL Image or a torch.Tensor')
    
    buffer = io.BytesIO()
    img.save(buffer, format="WEBP", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)


def butterworth_low_pass_filter(img, cutoff=40, order=2):
    """
    Apply Butterworth low-pass filter in frequency domain
    
    Args:
        img: PIL Image or torch.Tensor
        cutoff: Cutoff frequency for the filter
        order: Order of the Butterworth filter
    
    Returns:
        PIL Image after filtering
    """
    # Convert input to numpy array
    if isinstance(img, torch.Tensor):
        if img.dim() == 4:
            img = img.squeeze(0)
        img_np = img.detach().cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
        img_np = img_np.astype(float)
        img_np = np.clip(img_np, 0.0, 1.0)
    elif isinstance(img, Image.Image):
        img_np = np.array(img).astype(float) / 255.0
    else:
        raise TypeError('Input should be a PIL Image or torch.Tensor')

    filtered_img = np.zeros_like(img_np)
    rows, cols = img_np.shape[0], img_np.shape[1]
    crow, ccol = rows // 2, cols // 2
    
    # Create frequency grid
    y, x = np.mgrid[-crow:rows-crow, -ccol:cols-ccol]
    d = np.sqrt(x**2 + y**2)
    
    # Butterworth filter mask
    mask = 1 / (1 + (d / float(cutoff))**(2 * int(order)))

    # Apply filter to each channel
    for c in range(img_np.shape[2]):
        channel = img_np[:, :, c]
        f_transform = np.fft.fft2(channel)
        f_shift = np.fft.fftshift(f_transform)
        f_filtered = f_shift * mask
        f_inv = np.fft.ifftshift(f_filtered)
        img_filtered = np.real(np.fft.ifft2(f_inv))
        filtered_img[:, :, c] = img_filtered

    filtered_img = np.clip(filtered_img, 0, 1)
    filtered_uint8 = (filtered_img * 255).astype(np.uint8)
    return Image.fromarray(filtered_uint8)


def normalize_tensor(x):
    """
    Normalize tensor with ImageNet mean and std
    
    Args:
        x: Input tensor in range [0, 1]
    
    Returns:
        Normalized tensor
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def apply_freqpress_defense(tensor_img, mean, std, cutoff=40, order=2, quality=50):
    """
    Apply FreqPress defense (Butterworth + WebP) to a normalized tensor
    
    Args:
        tensor_img: [3,H,W] tensor, normalized with given mean/std
        mean: [3,1,1] normalization mean tensor
        std: [3,1,1] normalization std tensor
        cutoff: Butterworth cutoff frequency
        order: Butterworth filter order
        quality: WebP compression quality
    
    Returns:
        Defended tensor in normalized space
    """
    # Unnormalize
    img = tensor_img * std + mean
    img = img.clamp(0, 1).detach().cpu()
    
    # Convert to PIL
    pil = transforms.ToPILImage()(img)
    
    # Apply defenses
    pil = butterworth_low_pass_filter(pil, cutoff=cutoff, order=order)
    pil = webp_compression(pil, quality=quality)
    
    # Convert back to tensor
    t = transforms.ToTensor()(pil)
    
    # Renormalize
    mean_cpu = mean.detach().cpu()
    std_cpu = std.detach().cpu()
    t = (t - mean_cpu) / std_cpu
    
    return t.to(tensor_img.device)


def preprocess_batch_butterworth_webp(x, cutoff=40, order=2, quality=70):
    """
    Apply Butterworth + WebP preprocessing to a batch of images
    
    Args:
        x: Batch tensor [N,C,H,W] in range [0,1]
        cutoff: Butterworth cutoff frequency
        order: Butterworth filter order
        quality: WebP compression quality
    
    Returns:
        Processed and normalized batch tensor
    """
    out = []
    _to_tensor = transforms.ToTensor()
    
    for i in range(x.size(0)):
        pil = transforms.ToPILImage()(x[i].detach().cpu().clamp(0, 1))
        pil = butterworth_low_pass_filter(pil, cutoff=cutoff, order=order)
        pil = webp_compression(pil, quality=quality)
        out.append(_to_tensor(pil))
    
    batch = torch.stack(out, dim=0).to(x.device)
    return normalize_tensor(batch)


def save_unnormalized_tensor_image(tensor_img, mean, std, save_path):
    """
    Save a normalized image tensor as PNG, undoing normalization
    
    Args:
        tensor_img: [3,H,W] tensor on any device, normalized
        mean: [3,1,1] normalization mean
        std: [3,1,1] normalization std
        save_path: Path to save PNG file
    """
    img = tensor_img.detach().cpu()
    mean_cpu = mean.detach().cpu()
    std_cpu = std.detach().cpu()
    
    # Unnormalize
    img = img * std_cpu + mean_cpu
    img = img.clamp(0.0, 1.0)
    
    # Convert to PIL and save
    pil = transforms.ToPILImage()(img)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pil.save(save_path)
