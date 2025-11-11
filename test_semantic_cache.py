import numpy as np
from sematic_cache import SemanticCache, mean_pool_embeddings

def test_mean_pool_embeddings():
    patches = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    pooled = mean_pool_embeddings(patches)
    assert np.allclose(pooled, np.array([4.0, 5.0, 6.0]))
    print("mean_pool_embeddings: pooled correctly")

def test_get_on_empty_cache_adds_entry_and_returns_default():
    cache = SemanticCache(threshold=0.8)
    # create patch embeddings and compress (mean pool)
    patches = np.array([[0.1, 0.2, 0.3], [0.2, 0.1, 0.0]])
    emb = mean_pool_embeddings(patches)
    default_tokens = ["default-token"]
    result = cache.get(emb, default_tokens=default_tokens)
    assert result == default_tokens
    # cache should now contain one embedding
    assert len(cache.embeddings) == 1
    print("get on empty cache: added entry and returned default")

def test_put_and_retrieve_exact_match():
    cache = SemanticCache(threshold=0.9)
    patches = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
    emb = mean_pool_embeddings(patches)
    tokens = ["stored-token"]
    cache.put(emb, tokens)
    # retrieving the exact same vector should find a match
    retrieved = cache.get(emb)
    assert retrieved == tokens
    print("put and retrieve exact match: match found")

def test_retrieve_with_different_vector_misses_and_adds():
    cache = SemanticCache(threshold=0.99)  # high threshold to force misses for different vectors
    # first vector
    patches_a = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    emb_a = mean_pool_embeddings(patches_a)
    cache.put(emb_a, ["token-a"])
    # sufficiently different vector
    patches_b = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    emb_b = mean_pool_embeddings(patches_b)
    result_b = cache.get(emb_b, default_tokens=["default-b"])
    assert result_b == ["default-b"]
    # ensure a new entry was added
    assert len(cache.embeddings) >= 2
    print("different vector: no match, added new entry")

def test_zero_vector_handling():
    cache = SemanticCache(threshold=0.1)
    zero_emb = np.zeros(4)
    res = cache.get(zero_emb, default_tokens=["zero"])
    assert res == ["zero"]
    # ensure zero vector did not cause error and was added
    assert len(cache.embeddings) == 1
    print("zero vector: handled and added as entry")

def test_similar_match_within_threshold():  

    cache = SemanticCache(threshold=0.5)  # high threshold to force misses for different vectors
    # first vector
    patches_a = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    emb_a = mean_pool_embeddings(patches_a)
    cache.put(emb_a, ["token-a"])
    # sufficiently different vector
    patches_b = np.array([[0.5, 0.1, 0.5], [0.5, 0.1, 0.5]])
    emb_b = mean_pool_embeddings(patches_b)
    result_b = cache.get(emb_b, default_tokens=["default-b"])
    assert result_b == ["token-a"]
    # ensure a new entry was added
    assert len(cache.embeddings) == 1
    print("similar vector: matched within threshold")

if __name__ == "__main__":    
    test_mean_pool_embeddings()
    test_get_on_empty_cache_adds_entry_and_returns_default()
    test_put_and_retrieve_exact_match()
    test_retrieve_with_different_vector_misses_and_adds()
    test_zero_vector_handling()
    test_similar_match_within_threshold()