import subprocess
import sys
import os

def fix_and_train():
    print("🔧 Fixing environment and training model...")
    print("=" * 60)
    
    # Step 1: Check NumPy version
    print("\n📦 Checking NumPy version...")
    result = subprocess.run([sys.executable, "-c", "import numpy; print(numpy.__version__)"], 
                           capture_output=True, text=True)
    
    if "2." in result.stdout:
        print("❌ NumPy 2.x detected. Downgrading...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.24.3", "--force-reinstall"])
    
    # Step 2: Check OpenCV
    print("\n📦 Checking OpenCV...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python==4.8.0.76", "--force-reinstall"])
    
    # Step 3: Check TensorFlow
    print("\n📦 Checking TensorFlow...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.13.0", "--force-reinstall"])
    
    # Step 4: Verify imports
    print("\n✅ Verifying imports...")
    test_code = """
try:
    import numpy as np
    import cv2
    import tensorflow as tf
    print("✅ All imports successful!")
    print(f"NumPy: {np.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"TensorFlow: {tf.__version__}")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)
"""
    subprocess.check_call([sys.executable, "-c", test_code])
    
    # Step 5: Run training
    print("\n🚀 Starting training...")
    print("=" * 60)
    subprocess.check_call([sys.executable, "train_model.py"])

if __name__ == "__main__":
    fix_and_train()