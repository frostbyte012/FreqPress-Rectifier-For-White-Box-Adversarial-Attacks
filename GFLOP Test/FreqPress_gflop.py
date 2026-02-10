import numpy as np
import torch
from PIL import Image
import time
import torchvision
import io

def butterworth_low_pass_filter(img, cutoff=40, order=2):
    """
    Apply Butterworth Low Pass Filter to an image
    Parameters:
    - img: PIL Image or torch.Tensor
    - cutoff: Cutoff frequency (lower = more smoothing)
    - order: Filter order (higher = sharper transition)
    
    Returns:
    - PIL.Image: Filtered image
    """
    if isinstance(img, torch.Tensor):
        # Convert tensor to numpy array for processing
        is_tensor = True
        if img.dim() == 4:  # Remove batch dimension if present
            img = img.squeeze(0)
        
        # Store original device and shape
        original_device = img.device
        img_np = img.detach().cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    elif isinstance(img, Image.Image):
        is_tensor = False
        img_np = np.array(img).astype(float) / 255.0
    else:
        raise TypeError("Input should be a PIL Image or torch.Tensor")
    
    # Process each channel separately
    filtered_img = np.zeros_like(img_np)
    
    for c in range(img_np.shape[2]):
        # Get the current channel
        channel = img_np[:, :, c]
        
        # Apply FFT
        f_transform = np.fft.fft2(channel)
        f_transform_shifted = np.fft.fftshift(f_transform)
        
        # Create Butterworth filter mask
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create a grid of coordinates relative to the center
        y, x = np.mgrid[-crow:rows-crow, -ccol:cols-ccol]
        
        # Euclidean distance from center
        d = np.sqrt(x**2 + y**2)
        
        # Butterworth filter mask
        mask = 1 / (1 + (d / cutoff)**(2 * order))
        
        # Apply filter
        f_transform_filtered = f_transform_shifted * mask
        f_transform_filtered_back = np.fft.ifftshift(f_transform_filtered)
        
        # Inverse FFT
        img_filtered = np.real(np.fft.ifft2(f_transform_filtered_back))
        filtered_img[:, :, c] = img_filtered
    
    # Ensure values are within range [0, 1]
    filtered_img = np.clip(filtered_img, 0, 1)
    
    # Convert to PIL Image in all cases
    filtered_img_uint8 = (filtered_img * 255).astype(np.uint8)
    filtered_pil = Image.fromarray(filtered_img_uint8)
    
    # Return the PIL image
    return filtered_pil

def webp_compression(img, quality=50):
    if isinstance(img, torch.Tensor):
        # If it's a tensor, convert to PIL Image
        img = torchvision.transforms.ToPILImage()(img.cpu().squeeze().clamp(0, 1))
    elif not isinstance(img, Image.Image):
        raise TypeError("Input should be a PIL Image or a torch.Tensor")

    # Compress image using WebP
    buffer = io.BytesIO()
    img.save(buffer, format="WEBP", quality=quality)
    buffer.seek(0)

    # Open the compressed image
    webp_image = Image.open(buffer)

    return webp_image
    # Convert back to tensor
    #return transforms.ToTensor()(webp_image)


input_tensor = torch.randn(3, 224, 224)  # Example input tensor
# start = time.time()
# Call your target function or submodule here
intermediate = butterworth_low_pass_filter(input_tensor)
final = webp_compression(intermediate)
