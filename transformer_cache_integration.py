#!/usr/bin/env python3
"""
Integration of SemanticCache with Vision Transformer embeddings.
Demonstrates caching different types of embeddings from ViT models.
"""

import numpy as np
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
from sematic_cache import SemanticCache, mean_pool_embeddings
from typing import Dict, List, Optional, Tuple
import time

from semantic_scripts.compressor import PatchCompressor
from compress_vectors import get_image_embedding


class TransformerEmbeddingCache:
    """Enhanced semantic cache specifically designed for transformer embeddings."""
    
    def __init__(self, threshold: float = 0.8, model_name: str = 'google/vit-base-patch16-224'):
        # Caches keyed by embedding type names used elsewhere: 'pooled', 'cls_token', 'patch_pooled'
        self.pooled_cache = SemanticCache(threshold=threshold)
        self.patch_pooled_cache = SemanticCache(threshold=threshold)

        # Load ViT model and processor
        print(f"Loading model: {model_name}")
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name)

        self.embedding_layer = self.model.vit.embeddings
        _hidden_size = self.model.config.hidden_size
        _patch_size = getattr(self.model.config, 'patch_size', 16)

        self.compressor = PatchCompressor(embedding_layer=self.embedding_layer, in_channels=_hidden_size, compressed_channels=64)

        self.model.eval()
        self.compressor.eval()
    
    def extract_embeddings(self, image: Image.Image):
        """Extract different types of embeddings from an image."""
        
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            # Get Image Patch Embeddings
            compressed_vector = get_image_embedding(
                inputs,
                self.compressor,
                return_vector=True, device='gpu'
            )


        return compressed_vector
    
    def get_model_output(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs
    
    def get_image_output(self, image):
        compressed_vector = self.extract_embeddings(image)

        res = self.patch_pooled_cache.get(compressed_vector.numpy())
        hit = False

        if(res == []):
            res = self.get_model_output(image)
    
            self.patch_pooled_cache.put(compressed_vector.numpy(), res)
        else:
            # print("FOUND IN PATCH POOLED CACHE")
            hit = True

        return res, hit

    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about cached embeddings."""
        return {
            'pooled_entries': len(self.pooled_cache.embeddings),
            'patch_pooled_entries': len(self.patch_pooled_cache.embeddings),
            'total_entries': len(self.pooled_cache.embeddings) + len(self.patch_pooled_cache.embeddings)
        }


def load_sample_images() -> List[Tuple[Image.Image, List[str], str]]:
    """Load sample images for testing."""
    
    image_urls = [
        ('http://images.cocodataset.org/val2017/000000039769.jpg', ['cat', 'pet', 'animal'], 'cats_on_couch'),
        ('http://images.cocodataset.org/val2017/000000397133.jpg', ['person', 'human', 'people'], 'person_image'),
        ('http://images.cocodataset.org/val2017/000000037777.jpg', ['vehicle', 'transport'], 'vehicle_image'),
    ]
    
    images = []
    for url, tokens, image_id in image_urls:
        try:
            print(f"Loading image: {image_id}")
            image = Image.open(requests.get(url, stream=True).raw)
            images.append((image, tokens, image_id))
        except Exception as e:
            print(f"Failed to load {image_id}: {e}")
    
    return images


def demonstrate_cache_functionality():
    """Demonstrate the transformer embedding cache with real images."""
    
    print("=== Transformer Embedding Cache Demo ===\n")
    
    # Initialize cache
    cache_system = TransformerEmbeddingCache(threshold=0.7)
    
    # Load sample images
    print("Loading sample images...")
    sample_images = load_sample_images()
    
    if not sample_images:
        print("No images loaded. Please check your internet connection.")
        return
    
    print(f"Loaded {len(sample_images)} sample images.\n")
    
    # Cache the images
    print("Caching image embeddings...")
    for i, (image, tokens, image_id) in enumerate(sample_images):
        print(f"Caching image {i+1}: {image_id}")
        start_time = time.time()
        results = cache_system.get_image_output(image)
        elapsed = time.time() - start_time
        print(f"  Cached in {elapsed:.3f}s")
        # print(f"  Cached in {elapsed:.3f}s: {results}")
    
    print(f"\nCache stats: {cache_system.get_cache_stats()}\n")

    print("TESTING TO SEE IF IMAGES GOT CACHED...")
    for i, (image, tokens, image_id) in enumerate(sample_images):
        print(f"Caching image {i+1}: {image_id}")
        start_time = time.time()
        results = cache_system.get_image_output(image)
        elapsed = time.time() - start_time
        print(f"  Run in {elapsed:.3f}s")
        # print(f"  Cached in {elapsed:.3f}s: {results}")
    
    # # Test queries with the same images (should find matches)
    # print("=== Testing Queries ===")
    # for i, (query_image, original_tokens, image_id) in enumerate(sample_images):
    #     print(f"\nQuerying with image {i+1} ({image_id}):")
    #     start_time = time.time()
    #     results = cache_system.query_similar_images(query_image, ["query_default"])
    #     elapsed = time.time() - start_time
        
    #     print(f"  Query completed in {elapsed:.3f}s")
    #     for emb_type, (tokens, similarity) in results.items():
    #         print(f"  {emb_type}: similarity={similarity:.3f}, tokens={tokens[:3]}...")
    
    # # Test with a new image (should create new entries)
    # print(f"\n=== Testing New Image ===")
    # try:
    #     new_url = 'http://images.cocodataset.org/val2017/000000252219.jpg'
    #     new_image = Image.open(requests.get(new_url, stream=True).raw)
    #     print("Querying with completely new image:")
        
    #     results = cache_system.query_similar_images(new_image, ["new_image", "unknown"])
    #     for emb_type, (tokens, similarity) in results.items():
    #         print(f"  {emb_type}: similarity={similarity:.3f}, tokens={tokens}")
            
    # except Exception as e:
    #     print(f"Failed to test new image: {e}")
    
    # print(f"\nFinal cache stats: {cache_system.get_cache_stats()}")


def benchmark_embedding_types():
    """Compare different embedding types for similarity matching."""
    
    print("\n=== Embedding Type Benchmark ===")
    
    cache_system = TransformerEmbeddingCache(threshold=0.5)
    
    # Load one image and create slight variations
    try:
        url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        original_image = Image.open(requests.get(url, stream=True).raw)
        
        # Cache original
        cache_system.cache_image_embeddings(original_image, ["original", "cats"], "original")
        
        # Create a resized version (should be similar)
        resized_image = original_image.resize((200, 200))
        
        print("Comparing original vs resized image:")
        results = cache_system.query_similar_images(resized_image, ["resized"])
        
        for emb_type, (tokens, similarity) in results.items():
            match_status = "MATCH" if similarity >= 0.5 else "NEW"
            print(f"  {emb_type}: similarity={similarity:.3f} ({match_status})")
            
    except Exception as e:
        print(f"Benchmark failed: {e}")


if __name__ == "__main__":
    print("Starting Transformer-Cache Integration Demo")
    print("=" * 50)
    
    try:
        demonstrate_cache_functionality()
        # benchmark_embedding_types()
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()