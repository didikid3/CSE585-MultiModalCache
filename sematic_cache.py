import numpy as np
from typing import List, Dict, Optional
from sklearn.preprocessing import normalize



def mean_pool_embeddings(patch_embeddings: np.ndarray) -> np.ndarray:
    """Condense patch embeddings into a single vector using mean pooling."""
    return np.mean(patch_embeddings, axis=0)

class SemanticCache:
    def __init__(self, threshold: float = 0.5, normalize_embeddings: bool = True):
        self.cache: Dict[int, List] = {}  # Use integer keys instead of tuple keys
        self.embeddings: List[np.ndarray] = []
        self.threshold = threshold
        self.normalize_embeddings = normalize_embeddings
        self._next_id = 0
    
    def get_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        # Handle normalized vectors more efficiently
        if self.normalize_embeddings:
            # If vectors are already normalized, cosine similarity is just dot product
            similarity = np.dot(vec1, vec2)
        else:
            # Standard cosine similarity calculation
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
        
        return float(similarity)
    
    def get(self, embedding: np.ndarray, default_tokens: List = None) -> List:
        """Retrieve cached tokens using semantic similarity."""
        # Normalize embedding if required
        if self.normalize_embeddings:
            embedding = self._normalize_embedding(embedding)
            
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
            return self.cache[best_idx]
        else:
            # Create new cache entry
            self._add_entry(embedding, default_tokens or [])
            return default_tokens or []
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit vector."""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def _add_entry(self, embedding: np.ndarray, tokens: List) -> None:
        """Add a new entry to the cache."""
        # Normalize embedding if required
        if self.normalize_embeddings:
            embedding = self._normalize_embedding(embedding)
            
        # Use integer ID as key instead of tuple
        entry_id = self._next_id
        self.cache[entry_id] = tokens
        self.embeddings.append(embedding)
        self._next_id += 1
    
    def put(self, embedding: np.ndarray, tokens: List) -> None:
        """Manually add an entry to the cache."""
        self._add_entry(embedding, tokens)
    
    def get_best_match_info(self, embedding: np.ndarray) -> Optional[Dict]:
        """Get detailed information about the best match."""
        if self.normalize_embeddings:
            embedding = self._normalize_embedding(embedding)
            
        if len(self.embeddings) == 0:
            return None
            
        similarities = [self.get_similarity(embedding, cached) for cached in self.embeddings]
        max_similarity = max(similarities)
        best_idx = similarities.index(max_similarity)
        
        return {
            'similarity': max_similarity,
            'index': best_idx,
            'tokens': self.cache[best_idx],
            'is_match': max_similarity >= self.threshold
        }