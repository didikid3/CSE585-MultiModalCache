import numpy as np
from sematic_cache import SemanticCache, mean_pool_embeddings
from transformer_cache_integration import TransformerEmbeddingCache
import os
from pathlib import Path
import random
from PIL import Image

# Dataset paths
dataset_root = Path("chest_xray/test")
normal_dir = dataset_root / "NORMAL"
pneumonia_dir = dataset_root / "PNEUMONIA"

# Classification pipeline
def classification_pipeline(cache, image, label):
    
    # Image to Patch Embeddings
    compressed_emb = cache.extract_embeddings(image)
    # Patch Embeddings Compressiong


    # Cache Retrieval
    res = cache.pooled_cache.get(compressed_emb)
    # If Cache Miss, Put to Cache
    if(res == []):
        #res = cache.get_model_output(image)
        cache.pooled_cache.put(compressed_emb, [label])
        return False, False
    # If Cache Hit, Compare with Label
    else:
        cached_label = res[0]
        correct_classification = (cached_label == label)
        return True, correct_classification


if __name__ == "__main__":

    # Collect all image paths with labels
    print("Collecting image paths...")
    image_data = []
    for img_path in normal_dir.glob("*.jpeg"):
        image_data.append((str(img_path), "NORMAL"))
    for img_path in pneumonia_dir.glob("*.jpeg"):
        image_data.append((str(img_path), "PNEUMONIA"))

    # Shuffle the dataset
    print("Shuffling image data...")
    random.seed(67)
    random.shuffle(image_data)

    # Cache Initialization
    print("Initializing Transformer Embedding Cache...")
    cache = TransformerEmbeddingCache(threshold=0.8)

    # Process images
    num_images = len(image_data)
    num_cache_hits = 0
    num_correct_cache_hits = 0

    for img_path, label in image_data:
        #print(f"Processing image: {img_path} with label: {label}")
        image = Image.open(img_path).convert("RGB")
        #print(f"Image Shape: ", image.size)
        cache_hit, correct_classification = classification_pipeline(cache, image, label)
        if cache_hit:
            print(f"Cache hit! Correct classification: {correct_classification}")
            num_cache_hits += 1
            if correct_classification:
                num_correct_cache_hits += 1

    # Summary
    print("\n=== Evaluation Summary ===")
    print(f"Total images processed: {num_images}")
    print(f"Total cache hits: {num_cache_hits}")
    print(f"Total correct classifications from cache hits: {num_correct_cache_hits}")
    print(f"Cache Hit Rate: {num_cache_hits / num_images * 100:.2f}%")
    print(f"Correct Classification Rate from Cache Hits: {num_correct_cache_hits / num_cache_hits * 100:.2f}%" if num_cache_hits > 0 else "N/A")

    print("\nCache Statistics:")
    print(f"Number of cached entries: {len(cache.pooled_cache.cache)}")