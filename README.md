# Optimized Semantic Cache for Vision Transformers

## Overview
This project implements an optimized semantic cache system that avoids unnecessary Vision Transformer (ViT) computations by using fast image caching for exact matches and semantic similarity for approximate matches.

---

### Prerequisites
- **Python**: 3.11 or higher

### Step 1: Clone the Repository
```bash
git clone https://github.com/didikid3/CSE585-MultiModalCache.git
cd CSE585-MultiModalCache
```

### Step 2: Create a Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**Required Packages:**
- `numpy` - Numerical computing
- `torch` - PyTorch deep learning framework
- `transformers` - Hugging Face transformers library (for ViT models)
- `Pillow` - Image processing
- `scikit-learn` - Machine learning utilities (for normalization)
- `matplotlib` - Plotting library (for evaluation visualizations)
- `pandas` - Data manipulation and analysis
- `tabulate` - Pretty-printing tabular data

### Step 4: Run the Demo
```bash
# Run the optimized cache demo
python transformer_cache_integration.py
```

**Expected Output:**
```
Starting Transformer-Cache Integration Demo
==================================================
=== OPTIMIZED Transformer Embedding Cache Demo ===

Cache initialized with model: google/vit-base-patch16-224 (lazy loading)
Loading sample images...
Loading image: cats_on_couch
Loading image: person_image
Loading image: vehicle_image
...
```

### Step 5: Run Unit Tests
```bash
# Test the semantic cache implementation
python test_semantic_cache.py
```

**Expected Output:**
```
mean_pool_embeddings: pooled correctly
get on empty cache: added entry and returned default
put and retrieve exact match: match found
different vector: no match, added new entry
zero vector: handled and added as entry
similar vector: matched within threshold
...
```

### Step 6: Run Evaluation Benchmarks (Optional)

#### Chest X-Ray Classification Evaluation
```bash
# Evaluate cache performance on chest X-ray dataset
python xray_evaluation.py
```

This evaluates the cache on classifying chest X-rays (Normal vs Pneumonia) from the `chest_xray/` dataset. Results include:
- Cache hit rate across different similarity thresholds
- Classification accuracy when cache hits occur
- Performance metrics saved to `xray_eval_vit.csv`
- Visualization saved as `xray_threshold_evaluation.png`

#### LOTR Frame Classification Evaluation
```bash
# Evaluate cache performance on Lord of the Rings video frames
python lotr_evaluation.py
```

This evaluates the cache on classifying LOTR movie frames (character/scene detection) from the `lotr-frames/` dataset. Results include:
- Cache hit rate for video frame similarity
- Classification accuracy across frames
- Performance metrics saved to `lotr_eval_vit.csv`
- Visualization saved as `lotr_threshold_evaluation.png`

**Note:** These evaluations require the respective datasets:
- `chest_xray/test/` - Chest X-ray images (Normal and Pneumonia)
- `lotr-frames/` - LOTR video frames (ghandalf*, froto*, wide* patterns)

---

## Project Structure

```
CSE585-MultiModalCache/
├── sematic_cache.py                  # Base semantic cache implementation
├── transformer_cache_integration.py  # Optimized ViT cache system
├── test_semantic_cache.py           # Unit tests
├── xray_evaluation.py               # Chest X-ray classification benchmark
├── lotr_evaluation.py               # LOTR frame classification benchmark
├── compress_vectors.py              # Vector compression utilities
├── transformer.ipynb                # Jupyter notebook experiments
├── vision_transformer.py            # Simple ViT example
├── requirements.txt                 # Python dependencies
├── chest_xray/                      # X-ray dataset directory
├── lotr-frames/                     # LOTR frames dataset directory
├── images/                          # Additional image assets
├── xray_eval_vit.csv               # X-ray evaluation results
├── lotr_eval_vit.csv               # LOTR evaluation results
├── xray_threshold_evaluation.png   # X-ray performance visualization
├── lotr_threshold_evaluation.png   # LOTR performance visualization
└── README.md                        # This file
```

---

## Configuration Options

### Adjust Similarity Threshold
```python
# Lower threshold (0.5) = more lenient matching
cache = TransformerEmbeddingCache(threshold=0.5)

# Higher threshold (0.9) = stricter matching
cache = TransformerEmbeddingCache(threshold=0.9)
```

### Use Different ViT Models
```python
# Use a different pre-trained model
cache = TransformerEmbeddingCache(
    threshold=0.7,
    model_name='google/vit-large-patch16-224'
)
```

---

## Testing

### Run All Tests
```bash
python test_semantic_cache.py
```

### Test Individual Components
```python
# Test mean pooling
from sematic_cache import mean_pool_embeddings
import numpy as np

patches = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
pooled = mean_pool_embeddings(patches)
print(pooled)  # [2.5, 3.5, 4.5]
```

---

## Troubleshooting

### Issue: Model Download Fails
**Solution:** Ensure you have internet connectivity. The model (~330MB) will be downloaded on first use.

### Issue: Out of Memory
**Solution:** The ViT model requires ~2GB RAM. Close other applications or use a smaller model:
```python
cache = TransformerEmbeddingCache(model_name='google/vit-base-patch16-224-in21k')
```

---
