import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# The nested path structure found through folder scanning
COMPETITION_DIR = r'C:\Users\ibbi\Desktop\WEBCAM\dataset\the-silent-gap-asl-sign-language-challenge\competition_data\competition_data'
TRAIN_DIR = os.path.join(COMPETITION_DIR, 'train')

try:
    # Get sorted list of all classes (A-Z, space, del, nothing)
    classes = sorted(os.listdir(TRAIN_DIR))
    print(f"✅ Total Classes Found: {len(classes)}")
    
    # 1. CLASS DISTRIBUTION ANALYSIS
    class_counts = {}
    for c in classes:
        class_path = os.path.join(TRAIN_DIR, c)
        if os.path.isdir(class_path):
            class_counts[c] = len(os.listdir(class_path))

    # Plotting class distribution
    plt.figure(figsize=(15, 6))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), palette='viridis')
    plt.title('Images Per Class - Checking Imbalance', fontsize=14)
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.xlabel('Classes')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    # Printing raw numbers for precise analysis
    print("\n📊 Exact images per class:")
    for class_name, count in class_counts.items():
        print(f"Class {class_name}: {count} images")
        
    print(f"\n📈 Total Dataset Images: {sum(class_counts.values())}")
    
    # 2. SAMPLE IMAGE ANALYSIS (Checking Dimensions)
    sample_class = classes[0]
    sample_file = os.listdir(os.path.join(TRAIN_DIR, sample_class))[0]
    sample_img = cv2.imread(os.path.join(TRAIN_DIR, sample_class, sample_file))
    print(f"\n📏 Sample Image Shape (Height, Width, Channels): {sample_img.shape}")

except Exception as e:
    print("⚠️ An unexpected error occurred:", e)