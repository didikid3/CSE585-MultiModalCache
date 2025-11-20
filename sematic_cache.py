import numpy as np
from typing import List, Dict, Optional
from sklearn.preprocessing import normalize



def mean_pool_embeddings(patch_embeddings: np.ndarray) -> np.ndarray:
    """Condense patch embeddings into a single 1-D vector using mean pooling.

    This function is robust to inputs shaped as (n_patches, dim) or
    (batch, n_patches, dim) or other shapes where the last dimension is
    the embedding dimension. It averages across all axes except the
    last to produce a 1-D vector of length `dim`.
    """
    arr = np.asarray(patch_embeddings)
    if arr.ndim == 1:
        return arr
    # average across all axes except the last to return a 1-D vector
    axes = tuple(range(arr.ndim - 1))
    return np.mean(arr, axis=axes)

class SemanticCache:
    def __init__(self, threshold: float = 0.5, normalize_embeddings: bool = True):
        self.cache: Dict[int, List] = {}  # Use integer keys instead of tuple keys
        self.embeddings: List[np.ndarray] = []
        self.threshold = threshold
        self.normalize_embeddings = normalize_embeddings
        self._next_id = 0
    
    def _ensure_vector(self, embedding: np.ndarray) -> np.ndarray:
        """Ensure embedding is a 1-D vector of shape (dim,).

        If input has extra leading dimensions (e.g. batch or patch axes),
        average across them so the returned vector is length = last dim.
        """
        arr = np.asarray(embedding)
        if arr.ndim == 1:
            return arr
        axes = tuple(range(arr.ndim - 1))
        return np.mean(arr, axis=axes)

    def get_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Both inputs are coerced to 1-D vectors (averaging over leading
        dimensions if present) before similarity calculation.
        """
        v1 = self._ensure_vector(vec1)
        v2 = self._ensure_vector(vec2)

        # Handle normalized vectors more efficiently
        if self.normalize_embeddings:
            # If vectors are already normalized, cosine similarity is just dot product
            similarity = np.dot(v1, v2)
            return float(similarity)

        # Standard cosine similarity calculation
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def get(self, embedding: np.ndarray, default_tokens: Optional[List] = None) -> List:
        """Retrieve cached tokens using semantic similarity.

        If the cache is empty or no similarity exceeds the threshold and
        `default_tokens` is provided, the embedding and `default_tokens`
        will be added to the cache and `default_tokens` returned.
        """
        # Ensure embedding is 1-D vector
        embedding = self._ensure_vector(embedding)

        # Normalize embedding if required
        if self.normalize_embeddings:
            embedding = self._normalize_embedding(embedding)

        if len(self.embeddings) == 0:
            if default_tokens is not None:
                self._add_entry(embedding, default_tokens)
                return default_tokens
            return []

        # Find most similar cached embedding
        similarities = [self.get_similarity(embedding, cached) for cached in self.embeddings]
        max_similarity = max(similarities)

        if max_similarity >= self.threshold:
            # Return cached tokens
            best_idx = similarities.index(max_similarity)
            return self.cache[best_idx]

        # Miss: add default tokens if provided
        if default_tokens is not None:
            self._add_entry(embedding, default_tokens)
            return default_tokens

        return []
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit vector."""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def _add_entry(self, embedding: np.ndarray, tokens: List) -> None:
        """Add a new entry to the cache."""
        # Normalize embedding if required
        # ensure 1-D
        embedding = self._ensure_vector(embedding)
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
        # ensure 1-D and normalize if requested
        embedding = self._ensure_vector(embedding)
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