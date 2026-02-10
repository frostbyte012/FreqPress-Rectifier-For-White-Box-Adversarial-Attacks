"""
Utility operations for adversarial attacks
"""
import torch
import torch.nn.functional as F


def clamp(X, lower_limit, upper_limit):
    """
    Clamp tensor values between lower and upper limits
    
    Args:
        X: Input tensor
        lower_limit: Lower bound
        upper_limit: Upper bound
    
    Returns:
        Clamped tensor
    """
    return torch.max(torch.min(X, upper_limit), lower_limit)


def PCGrad(grad1, grad2, cos_sim, original_shape):
    """
    Projected Conflicting Gradient (PCGrad)
    Projects conflicting gradients to reduce interference
    
    Args:
        grad1: First gradient (flattened)
        grad2: Second gradient (flattened)
        cos_sim: Cosine similarity between gradients
        original_shape: Original shape to restore
    
    Returns:
        Projected gradient in original shape
    """
    # Project grad1 onto grad2 when they conflict (cos_sim < 0)
    projection = torch.where(
        cos_sim.view(-1, 1) < 0,
        grad1 - (torch.sum(grad1 * grad2, dim=1, keepdim=True) / 
                 (torch.sum(grad2 * grad2, dim=1, keepdim=True) + 1e-8)) * grad2,
        grad1
    )
    
    return projection.view(original_shape)
