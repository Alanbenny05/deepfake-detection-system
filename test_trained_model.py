import os
import cv2
import numpy as np
from tensorflow import keras

def test_model():
    print("🧪 Testing Trained Model")
    print("=" * 60)
    
    model_path = 'models/best_model.h5'
    if not os.path.exists(model_path):
        model_path = 'models/deepfake_detector.h5'
    
    if not os.path.exists(model_path):
        print("❌ Model not found! Please train the model first.")
        return
    
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
    
    test_dir = 'dataset/test'
    results = {'real': {'correct': 0, 'total': 0}, 
               'fake': {'correct': 0, 'total': 0}}
    
    for class_name in ['real', 'fake']:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        images = [f for f in os.listdir(class_dir) 
                 if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\n📁 Testing {class_name} images: {len(images)}")
        
        for img_name in images[:20]:
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            
            prediction = model.predict(img, verbose=0)
            prob_fake = float(prediction[0][0] * 100)
            prob_real = 100 - prob_fake
            
            predicted = 'fake' if prob_fake > prob_real else 'real'
            correct = predicted == class_name
            
            results[class_name]['total'] += 1
            if correct:
                results[class_name]['correct'] += 1
            
            print(f"  {img_name[:20]}: Predicted={predicted}, "
                  f"Real={prob_real:.1f}%, Fake={prob_fake:.1f}% "
                  f"{'✅' if correct else '❌'}")
    
    print("\n📊 Summary:")
    for class_name in ['real', 'fake']:
        if results[class_name]['total'] > 0:
            acc = results[class_name]['correct'] / results[class_name]['total'] * 100
            print(f"  {class_name}: {results[class_name]['correct']}/{results[class_name]['total']} "
                  f"({acc:.1f}%)")

if __name__ == "__main__":
    test_model()