import torch.nn as nn
import math

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
            nn.ReLU(),  # Maybe switch to swiglu, see if there is change in perf?
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
