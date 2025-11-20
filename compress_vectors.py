from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import pandas as pd
from tabulate import tabulate

# Cosine Simlilarity and L2
from semantic_scripts.compare import compare_images
from semantic_scripts.compressor import PatchCompressor


def get_image_embedding(image_path: str, 
                        processor: ViTImageProcessor,
                        compressor: PatchCompressor,
                        return_vector=True, device=None
    ) -> torch.Tensor:
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
    for i in range(len(image_paths)):
        for j in range(i + 1, len(image_paths)):
            img1 = image_paths[i]
            img2 = image_paths[j]
            e1 = get_image_embedding(img1, processor=processor, compressor=compressor, return_vector=True)
            e2 = get_image_embedding(img2, processor=processor, compressor=compressor, return_vector=True)

            cos = compare_images(e1, e2, method='cosine')
            l2 = compare_images(e1, e2, method='l2')
            dot = compare_images(e1, e2, method='dot')

            sims.append({
                'img1': img1,
                'img2': img2,
                'cosine': float(cos),
                'l2': float(l2),
                'dot': float(dot),
            })

    sims_df = pd.DataFrame(sims)

    print(tabulate(
        sims_df.to_numpy(), 
        headers=sims_df.columns.to_list(), 
        showindex=False
    ))

    


