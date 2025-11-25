import numpy as np
from sematic_cache import SemanticCache, mean_pool_embeddings
from transformer_cache_integration import TransformerEmbeddingCache
import os
from pathlib import Path
import random
from PIL import Image
import sys
import pandas as pd 
import matplotlib.pyplot as plt    
import csv
import time

np.random.seed(67)

# Dataset paths
dataset_root = Path("chest_xray/test")
normal_dir = dataset_root / "NORMAL"
pneumonia_dir = dataset_root / "PNEUMONIA"

# Classification pipeline
def classification_pipeline(cache, image, label):
    
    # Image to Patch Embeddings
    compressed_emb = cache.extract_embeddings(image)
    # Patch Embeddings Compression


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

def get_images(seed):
    # Collect all image paths with labels
    print("Collecting image paths...")
    image_data = []
    for img_path in normal_dir.glob("*.jpeg"):
        image_data.append((str(img_path), "NORMAL"))
    for img_path in pneumonia_dir.glob("*.jpeg"):
        image_data.append((str(img_path), "PNEUMONIA"))

    # Shuffle the dataset
    print("Shuffling image data...")
    random.seed(seed)
    random.shuffle(image_data)
    return image_data

def write_results_to_csv(threshold, hit_rate, correct_rate, time_taken):

    file_exists = os.path.isfile('xray_evaluation_time_results.csv')
    with open('xray_evaluation_time_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Threshold', 'Cache Hit Rate', 'Correct Classification Rate from Cache Hits', 'Time Taken'])
        writer.writerow([threshold, hit_rate, correct_rate, time_taken])

def evaluate(threshold, verbose=True):

    # Cache Initialization
    print("Initializing Transformer Embedding Cache...")
    cache = TransformerEmbeddingCache(threshold=threshold)

    start_time = time.perf_counter()

    image_data = get_images(seed=67)

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
            if verbose: print(f"Cache hit! Correct classification: {correct_classification}")
            num_cache_hits += 1
            if correct_classification:
                num_correct_cache_hits += 1


    end_time = time.perf_counter()

    time_taken = end_time - start_time
    correct_classification_rate = (num_correct_cache_hits / num_cache_hits) if num_cache_hits > 0 else np.nan
    hit_rate = (num_cache_hits / num_images) if num_images > 0 else np.nan

    time_taken = round(time_taken, 3)
    correct_classification_rate = round(correct_classification_rate, 3) if not np.isnan(correct_classification_rate) else np.nan
    hit_rate = round(hit_rate, 3) if not np.isnan(hit_rate) else np.nan

    # Summary
    if verbose:
        print("\n=== Evaluation Summary ===")
        print(f"Time through all images: {time_taken:.2f} seconds\n\n")
        print(f"Total images processed: {num_images}")
        print(f"Total cache hits: {num_cache_hits}")
        print(f"Total correct classifications from cache hits: {num_correct_cache_hits}")
        print(f"Cache Hit Rate: {hit_rate * 100:.2f}%")
        print(f"Correct Classification Rate from Cache Hits: {correct_classification_rate * 100:.2f}%")

        print("\nCache Statistics:")
        print(f"Number of cached entries: {len(cache.pooled_cache.cache)}")

    # Write results to CSV
    print("Writing results to CSV...")
    write_results_to_csv(threshold, hit_rate, correct_classification_rate, time_taken)

def plot_thresholds():

    df = pd.read_csv('xray_evaluation_time_results.csv')
    df = df.sort_values(by='Threshold')

    fig, ax1 = plt.subplots(figsize=(8,5))

    # Left y-axis (Hit Rate + Correct Rate)
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Hit Rate / Correct Rate")
    ax1.set_ylim(0, 1.05)

    ax1.plot(df['Threshold'], df['Cache Hit Rate'], color='blue', label="Cache Hit Rate")
    ax1.plot(df['Threshold'], df['Correct Classification Rate from Cache Hits'], color='green', label="Correct Classification Rate")

    # Right y-axis (Time Taken)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Time Elapsed during Sequence (seconds)")
    ax2.set_ylim(0, 31)

    ax2.plot(df['Threshold'], df['Time Taken'], linestyle='--', color='gray', label="Time Taken")

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    plt.title("Threshold Evalutaion on XRay Dataset")
    #plt.tight_layout()
    plt.savefig("xray_threshold_evaluation.png")   


if __name__ == "__main__":


    if len(sys.argv) > 2:
        min_threshold = float(sys.argv[1])
        max_threshold = float(sys.argv[2])
        step = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
        thresholds = np.arange(min_threshold, max_threshold, step)
        for th in thresholds:
            print(f"\n=== Evaluating Threshold ===")
            print(f"========== {f'{th:.4f}'} ==========")
            evaluate(round(th,4), verbose=False)

        plot_thresholds()
    elif len(sys.argv) > 1:
        th = float(sys.argv[1])
        evaluate(th, verbose=True)
    else:
        plot_thresholds()
    
    

    