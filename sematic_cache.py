import numpy as np
from typing import List, Dict
from sklearn.preprocessing import normalize



def mean_pool_embeddings(patch_embeddings: np.ndarray) -> np.ndarray:
    """Condense patch embeddings into a single vector using mean pooling."""
    return np.mean(patch_embeddings, axis=0)

class SemanticCache:
    def __init__(self, threshold: float = 0.5):
        self.cache: Dict[tuple, List] = {}
        self.embeddings: List[np.ndarray] = []
        self.threshold = threshold
    
    def get_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        print(f"Computed similarity: {similarity}")
        return similarity
    
    def get(self, embedding: np.ndarray, default_tokens: List = None) -> List:
        """Retrieve cached tokens using semantic similarity."""
        if len(self.embeddings) == 0:
            # No cache entries, create new one
            self._add_entry(embedding, default_tokens or [])
            return default_tokens or []
        
        # Find most similar cached embedding
        similarities = [self.get_similarity(embedding, cached) for cached in self.embeddings]
        max_similarity = max(similarities)
        
        if max_similarity >= self.threshold:
            # Return cached tokens
            best_idx = similarities.index(max_similarity)
            key = tuple(self.embeddings[best_idx])
            return self.cache[key]
        else:
            # Create new cache entry
            self._add_entry(embedding, default_tokens or [])
            return default_tokens or []
    
    def _add_entry(self, embedding: np.ndarray, tokens: List) -> None:
        """Add a new entry to the cache."""
        key = tuple(embedding)
        self.cache[key] = tokens
        self.embeddings.append(embedding)
    
    def put(self, embedding: np.ndarray, tokens: List) -> None:
        """Manually add an entry to the cache."""
        self._add_entry(embedding, tokens)