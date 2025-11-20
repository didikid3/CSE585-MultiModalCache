# Similarity helpers: cosine, L2 and a convenience compare_images() function
def cosine_similarity(a, b, eps=1e-8):
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

def l2_distance(a, b):
    """Compute L2 (Euclidean) distance between two vectors or batches."""
    a = a.squeeze()
    b = b.squeeze()
    if a.ndim == 1:
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
    d = (a - b).norm(p=2, dim=1)
    return d.item() if d.numel() == 1 else d
