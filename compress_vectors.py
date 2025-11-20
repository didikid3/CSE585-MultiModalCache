from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import torch.nn as nn
import math
import pandas as pd
import numpy as np
from tabulate import tabulate

# Cosine Simlilarity and L2
from semantic_scripts.compare import *

class PatchCompressor(nn.Module):
    """Compress ViT patch embeddings (C x Gh x Gw) -> (C' x Gh x Gw).
    This module can accept either precomputed patch embeddings or raw ViT pixel inputs
    and will use the provided `embedding_layer` to compute patches when given pixel tensors."""
    def __init__(self, embedding_layer=None, in_channels=None, compressed_channels=64):
        super().__init__()
        # embedding_layer: the ViT embeddings module (full_model.vit.embeddings)
        self.embedding_layer = embedding_layer
        # determine in_channels from either provided value or embedding_layer config
        if in_channels is None and embedding_layer is not None:
            try:
                in_channels = embedding_layer.position_embeddings.shape[-1]
            except Exception:
                # fall back to None — user should provide in_channels
                in_channels = None
        if in_channels is None:
            raise ValueError('in_channels must be provided if embedding_layer does not expose channel size')

        # 1x1 convs act as channel compressors while preserving spatial layout
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, compressed_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(compressed_channels, compressed_channels, kernel_size=1),
        )

    def _patch_embeddings_from_pixels(self, pixel_values):
        # Use the supplied embedding_layer to obtain patch embeddings from pixels
        # embedding_layer.patch_embeddings should accept pixel_values and return either
        # (B, N, C) or (B, C, Gh, Gw). We normalize to (B, C, Gh, Gw).
        patch_emb = self.embedding_layer.patch_embeddings(pixel_values)
        if patch_emb.ndim == 3:
            B, N, C = patch_emb.shape
            G = int(math.sqrt(N))
            patch_emb = patch_emb.transpose(1, 2).reshape(B, C, G, G)
        elif patch_emb.ndim == 4:
            pass
        else:
            raise ValueError(f'Unexpected patch_emb shape: {patch_emb.shape}')
        return patch_emb

    def forward(self, x, is_pixel_values=False):
        """If `is_pixel_values` is True, `x` should be pixel_values (B,3,H,W),
        otherwise `x` should be patch embeddings (B,N,C) or (B,C,Gh,Gw).
        Returns compressed feature map with shape (B, C', Gh, Gw)."""
        if is_pixel_values:
            if self.embedding_layer is None:
                raise RuntimeError('No embedding_layer available to compute patch embeddings from pixels')
            patch_emb = self._patch_embeddings_from_pixels(x)
        else:
            patch_emb = x
            if patch_emb.ndim == 3:
                B, N, C = patch_emb.shape
                G = int(math.sqrt(N))
                patch_emb = patch_emb.transpose(1, 2).reshape(B, C, G, G)
            elif patch_emb.ndim == 4:
                pass
            else:
                raise ValueError(f'Unexpected patch_emb shape: {patch_emb.shape}')
        return self.net(patch_emb)


def get_image_embedding(image_path, return_vector=True, device=None):
    """Load an image, compute ViT patch embeddings (via compressor), compress with a small CNN,
    and return either the compressed feature map or flattened vector."""

    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors='pt')
    pixel_values = inputs.get('pixel_values')  # (B, 3, H, W)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pixel_values = pixel_values.to(device)
    compressor.to(device)

    compressed_map = compressor(pixel_values, is_pixel_values=True)

    if return_vector:
        # flatten spatial grid into a single vector per image
        return compressed_map.flatten(start_dim=1).cpu()
    return compressed_map.cpu()


def compare_images(path1, path2, method='cosine', device=None):
    """Compute similarity between two image paths using compressed embeddings."""
    # Obtain flattened compressed embeddings (shape: (1, D) or (D,))
    e1 = get_image_embedding(path1, return_vector=True, device=device)
    e2 = get_image_embedding(path2, return_vector=True, device=device)

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




processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
full_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# embedding_layer exposes the ViT embedding utilities (patch projection, cls token, etc.)
embedding_layer = full_model.vit.embeddings

hidden_size = full_model.config.hidden_size
patch_size = getattr(full_model.config, 'patch_size', 16)

# TODO: Adjust compressor amount as we want
compressor = PatchCompressor(embedding_layer=embedding_layer, in_channels=hidden_size, compressed_channels=64)


full_model.eval()
compressor.eval()


image_paths = [
    "images/lotr/img (12).jpg",
    "images/lotr/img (100).jpg",
    "images/lotr/img (101).jpg",
]

sims = []
with torch.no_grad():
    for image in image_paths:
        vector = get_image_embedding(image, return_vector=True)
        print(f'Image: {image[:10]}, Compressed vector shape: {vector.shape}')

    for i in range(len(image_paths)):
        for j in range(i + 1, len(image_paths)):
            img1 = image_paths[i]
            img2 = image_paths[j]
            cos = compare_images(img1, img2, method='cosine')
            l2 = compare_images(img1, img2, method='l2')
            dot = compare_images(img1, img2, method='dot')
            sims.append({
                'img1': img1,
                'img2': img2,
                'cosine': float(cos),
                'l2': float(l2),
                'dot': float(dot),
            })

    sims_df = pd.DataFrame(sims)


    print(tabulate(sims_df, headers='keys', 
                tablefmt='psql', 
                showindex=False
    ))

    


