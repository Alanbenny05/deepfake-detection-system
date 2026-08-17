import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_sample_data():
    """Create sample data for quick testing"""
    print("Creating sample dataset for quick training...")
    
    # Create directories
    for split in ['train', 'validation', 'test']:
        for cls in ['real', 'fake']:
            os.makedirs(f'dataset/{split}/{cls}', exist_ok=True)
    
    # Generate synthetic images (for demonstration only)
    # In reality, you should use real images
    for i in range(100):
        # Real image (random noise with some pattern)
        img = np.random.rand(224, 224, 3) * 255
        cv2.imwrite(f'dataset/train/real/real_{i}.jpg', img.astype(np.uint8))
        
        # Fake image (different pattern)
        img = np.random.rand(224, 224, 3) * 255
        cv2.imwrite(f'dataset/train/fake/fake_{i}.jpg', img.astype(np.uint8))
    
    for i in range(20):
        img = np.random.rand(224, 224, 3) * 255
        cv2.imwrite(f'dataset/validation/real/real_{i}.jpg', img.astype(np.uint8))
        img = np.random.rand(224, 224, 3) * 255
        cv2.imwrite(f'dataset/validation/fake/fake_{i}.jpg', img.astype(np.uint8))
    
    print("Sample dataset created!")

if __name__ == "__main__":
    create_sample_data()
    print("Now run: python train_model.py")