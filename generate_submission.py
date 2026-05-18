import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report

# Paths
BASE_PATH = r'C:\Users\ibbi\Desktop\WEBCAM\dataset\the-silent-gap-asl-sign-language-challenge'
COMPETITION_DIR = os.path.join(BASE_PATH, 'competition_data', 'competition_data')
TRAIN_DIR = os.path.join(COMPETITION_DIR, 'train')

CLASSES_PATH = r'C:\Users\ibbi\Desktop\WEBCAM\dataset\classes.txt'
MODEL_PATH = r'C:\Users\ibbi\Desktop\WEBCAM\dataset\alphabet_cnn_model.keras'

print("="*60)
print("🔍 ASL MODEL EVALUATION (Using Train Set)")
print("="*60 + "\n")

# Load classes
print("📂 Loading classes...")
with open(CLASSES_PATH, 'r') as f:
    classes_raw = f.read().strip()
    if ',' in classes_raw:
        classes = [c.strip() for c in classes_raw.split(',')]
    else:
        classes = [c.strip() for c in classes_raw.split('\n')]

print(f"✅ {len(classes)} classes loaded\n")

# Load model
print("🤖 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded!\n")

# Calculate accuracy on train set
print("="*60)
print("🔍 CALCULATING ACCURACY ON TRAIN SET")
print("="*60 + "\n")

y_true = []
y_pred = []
total_images = 0
correct_total = 0

for class_name in classes:
    class_dir = os.path.join(TRAIN_DIR, class_name)
    
    if not os.path.exists(class_dir):
        print(f"⚠️  Folder not found: {class_name}")
        continue
    
    class_images = os.listdir(class_dir)
    total_images += len(class_images)
    print(f"📁 Processing {class_name}: {len(class_images)} images...", end=" ")
    
    correct_for_class = 0
    
    for img_file in class_images:
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img_path = os.path.join(class_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        # Preprocessing
        img_resized = cv2.resize(img, (64, 64))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img_rgb, axis=0).astype(np.float32) / 255.0
        
        # Predict
        output_data = model.predict(input_data, verbose=0)
        pred_class = classes[np.argmax(output_data)]
        
        y_true.append(class_name)
        y_pred.append(pred_class)
        
        if pred_class == class_name:
            correct_for_class += 1
            correct_total += 1
    
    class_accuracy = (correct_for_class / len(class_images)) * 100 if class_images else 0
    print(f"✅ {class_accuracy:.1f}%")

# Overall accuracy
if len(y_true) > 0:
    accuracy = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*60)
    print(f"✅ FINAL TRAIN SET ACCURACY: {accuracy * 100:.2f}%")
    print(f"   Total predictions: {len(y_true)}")
    print(f"   Correct predictions: {correct_total}")
    print(f"   Wrong predictions: {len(y_true) - correct_total}")
    print("="*60 + "\n")
    
    # Detailed report
    print("📊 CLASSIFICATION REPORT:\n")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    print("="*60)
    print("✅ EVALUATION COMPLETE!")
    print("="*60)
else:
    print("\n❌ No images were processed!")