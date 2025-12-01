
from transformer_cache_integration import TransformerEmbeddingCache
from transformers import ViTForImageClassification

import numpy as np
import os
from pathlib import Path
import random
from PIL import Image
import sys
import pandas as pd 
import matplotlib.pyplot as plt    
import csv
import time
import tqdm


seed = 1234
np.random.seed(seed)
random.seed(seed)

model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
id2label = model.config.id2label
label2id = model.config.label2id

dataset_root = Path("image_net/ImageNetV2-threshold-0.7")
image_data = []
for class_dir in dataset_root.iterdir():
    if class_dir.is_dir():
        label = class_dir.name
        for img_path in class_dir.glob("*.jpeg"):
            image_data.append((str(img_path), int(label)))
random.shuffle(image_data)

def evaluate(threshold):
    cache = TransformerEmbeddingCache(threshold=threshold)
    
    total_images = len(image_data)
    cache_hits = 0
    correct_classifications = 0

    start_time = time.time()

    for img_path, label in tqdm.tqdm(image_data, desc="Evaluating Images", unit="image"):
        image = Image.open(img_path).convert("RGB")
        res, hit = cache.get_image_output(image)
        
        if hit == True:
            cache_hits += 1
            logits = res.logits
            predicted_class_idx = logits.argmax(-1).item()
            if predicted_class_idx == label:
                correct_classifications += 1
            
        

    end_time = time.time()
    time_taken = end_time - start_time

    hit_rate = cache_hits / total_images
    correct_rate = correct_classifications / cache_hits

    return hit_rate, correct_rate, time_taken

def write_results_to_csv(threshold, hit_rate, correct_rate, time_taken):

    file_exists = os.path.isfile('image_net_evaluation_time_results.csv')
    with open('image_net_evaluation_time_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Threshold', 'Hit Rate', 'Correct Rate', 'Time Taken (s)'])
        writer.writerow([threshold, hit_rate, correct_rate, time_taken])

if __name__ == "__main__":
    thresholds = [round(0.5 + i*0.05, 2) for i in range(9)]
    thresholds.extend([round(0.65 + i*0.02, 2) for i in range(13)])
    for threshold in tqdm.tqdm(thresholds, desc="Thresholds", unit="threshold"):
        hit_rate, correct_rate, time_taken = evaluate(threshold)
        write_results_to_csv(threshold, hit_rate, correct_rate, time_taken)

