import torch
from typing import Union

def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps=1e-8):
    """Compute cosine similarity between two vectors or batches."""
    # accept 1D or 2D tensors; return scalar for single pair or 1D tensor for batch
    a = a.squeeze()
    b = b.squeeze()
    if a.ndim == 1:
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
    a = a.float()
    b = b.float()
    a_norm = a / (a.norm(dim=1, keepdim=True) + eps)
    b_norm = b / (b.norm(dim=1, keepdim=True) + eps)
    sims = (a_norm * b_norm).sum(dim=1)
    return sims.item() if sims.numel() == 1 else sims

def l2_distance(a: torch.Tensor, b: torch.Tensor):
    """Compute L2 (Euclidean) distance between two vectors or batches."""
    a = a.squeeze()
    b = b.squeeze()
    if a.ndim == 1:
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
    d = (a - b).norm(p=2, dim=1)
    return d.item() if d.numel() == 1 else d

def compare_images(e1: torch.Tensor, e2: torch.Tensor,
                   method='cosine',
                   device=None) -> Union[float, torch.Tensor]:
    """Compute similarity between two image paths using compressed embeddings."""


    if method == 'cosine':
        return cosine_similarity(e1, e2)
    elif method == 'l2':
        return l2_distance(e1, e2)
    elif method == 'dot':
        v1 = e1.squeeze()
        v2 = e2.squeeze()
        if v1.ndim == 2:
            v1 = v1.squeeze(0)
            v2 = v2.squeeze(0)
        return (v1 * v2).sum().item()
    else:
        raise ValueError(f'Unknown method: {method}')
