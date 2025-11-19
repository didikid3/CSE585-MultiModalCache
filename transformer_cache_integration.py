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


class TransformerEmbeddingCache:
    """Enhanced semantic cache specifically designed for transformer embeddings."""
    
    def __init__(self, threshold: float = 0.8, model_name: str = 'google/vit-base-patch16-224'):
        # Caches keyed by embedding type names used elsewhere: 'pooled', 'cls_token', 'patch_pooled'
        self.pooled_cache = SemanticCache(threshold=threshold)
        self.cls_token_cache = SemanticCache(threshold=threshold)
        self.patch_pooled_cache = SemanticCache(threshold=threshold)

        # Load ViT model and processor
        print(f"Loading model: {model_name}")
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name)
        self.model.eval()

        print(f"Model loaded successfully!")
        print(f"Hidden dimension: {self.model.config.hidden_size}")
        print(f"Image size: {self.model.config.image_size}")
        print(f"Patch size: {self.model.config.patch_size}")
    
    def extract_embeddings(self, image: Image.Image) -> Dict[str, np.ndarray]:
        """Extract different types of embeddings from an image."""
        
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            # Get ViT outputs
            vit_outputs = self.model.vit(**inputs)
            
            # 1. Pooled embedding (global image representation)
            pooled = getattr(vit_outputs, "pooler_output", None)
            if pooled is None:
                # Use CLS token if no pooler output
                pooled = vit_outputs.last_hidden_state[:, 0, :]
            pooled_np = pooled.cpu().numpy().flatten()
            
            # 2. CLS token embedding (first token of sequence)
            cls_token = vit_outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            
            # 3. Patch embeddings (mean-pooled from all patch tokens, excluding CLS)
            patch_tokens = vit_outputs.last_hidden_state[:, 1:, :]  # Exclude CLS token
            patch_embeddings = patch_tokens.cpu().numpy().squeeze(0)  # Remove batch dim
            patch_pooled = mean_pool_embeddings(patch_embeddings)
            
            return {
                'pooled': pooled_np,
                'cls_token': cls_token,
                'patch_pooled': patch_pooled,
                'full_sequence': vit_outputs.last_hidden_state.cpu().numpy().squeeze(0)  # Keep for analysis
            }
    
    def cache_image_embeddings(self, image: Image.Image, tokens: List[str], image_id: str = None) -> Dict[str, bool]:
        """Cache embeddings for an image with associated tokens."""
        
        embeddings = self.extract_embeddings(image)
        
        # Cache different embedding types
        results = {}
        
        # Add image_id to tokens if provided
        enhanced_tokens = tokens.copy()
        if image_id:
            enhanced_tokens.append(f"image_id:{image_id}")
        
        self.pooled_cache.put(embeddings['pooled'], enhanced_tokens)
        self.cls_token_cache.put(embeddings['cls_token'], enhanced_tokens)
        self.patch_pooled_cache.put(embeddings['patch_pooled'], enhanced_tokens)
        
        results['pooled'] = True
        results['cls_token'] = True
        results['patch_pooled'] = True
        
        return results
    
    def query_similar_images(self, query_image: Image.Image, default_tokens: List[str] = None) -> Dict[str, Tuple[List[str], float]]:
        """Query for similar images using different embedding types."""
        
        embeddings = self.extract_embeddings(query_image)
        results = {}
        
        if default_tokens is None:
            default_tokens = ["unknown_image"]
        
        # Query each cache type and return both tokens and similarity
        for emb_type, embedding in [
            ('pooled', embeddings['pooled']),
            ('cls_token', embeddings['cls_token']),
            ('patch_pooled', embeddings['patch_pooled'])
        ]:
            cache = getattr(self, f"{emb_type}_cache")
            
            # Get cached result
            cached_tokens = cache.get(embedding, default_tokens.copy())
            
            # Calculate best similarity if cache has entries
            best_similarity = 0.0
            if len(cache.embeddings) > 0:
                similarities = [cache.get_similarity(embedding, cached) for cached in cache.embeddings]
                best_similarity = max(similarities)
            
            results[emb_type] = (cached_tokens, best_similarity)
        
        return results
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about cached embeddings."""
        return {
            'pooled_entries': len(self.pooled_cache.embeddings),
            'cls_token_entries': len(self.cls_token_cache.embeddings),
            'patch_pooled_entries': len(self.patch_pooled_cache.embeddings),
            'total_entries': len(self.pooled_cache.embeddings) + len(self.cls_token_cache.embeddings) + len(self.patch_pooled_cache.embeddings)
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
        results = cache_system.cache_image_embeddings(image, tokens, image_id)
        elapsed = time.time() - start_time
        print(f"  Cached in {elapsed:.3f}s: {results}")
    
    print(f"\nCache stats: {cache_system.get_cache_stats()}\n")
    
    # Test queries with the same images (should find matches)
    print("=== Testing Queries ===")
    for i, (query_image, original_tokens, image_id) in enumerate(sample_images):
        print(f"\nQuerying with image {i+1} ({image_id}):")
        start_time = time.time()
        results = cache_system.query_similar_images(query_image, ["query_default"])
        elapsed = time.time() - start_time
        
        print(f"  Query completed in {elapsed:.3f}s")
        for emb_type, (tokens, similarity) in results.items():
            print(f"  {emb_type}: similarity={similarity:.3f}, tokens={tokens[:3]}...")
    
    # Test with a new image (should create new entries)
    print(f"\n=== Testing New Image ===")
    try:
        new_url = 'http://images.cocodataset.org/val2017/000000252219.jpg'
        new_image = Image.open(requests.get(new_url, stream=True).raw)
        print("Querying with completely new image:")
        
        results = cache_system.query_similar_images(new_image, ["new_image", "unknown"])
        for emb_type, (tokens, similarity) in results.items():
            print(f"  {emb_type}: similarity={similarity:.3f}, tokens={tokens}")
            
    except Exception as e:
        print(f"Failed to test new image: {e}")
    
    print(f"\nFinal cache stats: {cache_system.get_cache_stats()}")


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
        benchmark_embedding_types()
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()