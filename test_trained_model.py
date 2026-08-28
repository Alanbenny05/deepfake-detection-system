import os
import cv2
import numpy as np

def test_model():
    print("[*] Testing Trained Model")
    print("=" * 60)
    
    candidate_paths = [
        'models/best_model.h5',
        'models/deepfake_detector.h5',
        'models/deepfake_model.h5'
    ]
    
    model_path = None
    for path in candidate_paths:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            model_path = path
            break
    
    if not model_path:
        print("[!] No trained .h5 model file found with non-zero size.")
        print("[*] Using DeepfakeDetector engine for test evaluation...")
        from detector.ml_model import detector
        use_detector_engine = True
    else:
        try:
            from tensorflow import keras
            model = keras.models.load_model(model_path)
            print(f"[+] Model loaded successfully from {model_path}!")
            use_detector_engine = False
        except Exception as e:
            print(f"[!] Could not load model via TensorFlow ({e}).")
            print("[*] Using DeepfakeDetector engine for test evaluation...")
            from detector.ml_model import detector
            use_detector_engine = True
    
    test_dir = 'dataset/test'
    if not os.path.exists(test_dir):
        print(f"[X] Test directory '{test_dir}' not found! Run python generate_sample_dataset.py first.")
        return
        
    results = {'real': {'correct': 0, 'total': 0}, 
               'fake': {'correct': 0, 'total': 0}}
    
    for class_name in ['real', 'fake']:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        images = [f for f in os.listdir(class_dir) 
                 if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\n[+] Testing {class_name.upper()} images: {len(images)}")
        
        for img_name in images[:20]:
            img_path = os.path.join(class_dir, img_name)
            
            if use_detector_engine:
                pred = detector.predict_image(img_path)
                predicted = pred['result']
                prob_real = pred['probability_real']
                prob_fake = pred['probability_fake']
            else:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=0)
                
                prediction = model.predict(img, verbose=0)
                prob_fake = float(prediction[0][0] * 100)
                prob_real = 100 - prob_fake
                predicted = 'fake' if prob_fake > prob_real else 'real'
            
            correct = (predicted == class_name)
            
            results[class_name]['total'] += 1
            if correct:
                results[class_name]['correct'] += 1
            
            status_symbol = "[OK]" if correct else "[MISMATCH]"
            print(f"  {img_name[:25]:<25}: Predicted={predicted:<4} | "
                  f"Real={prob_real:.1f}%, Fake={prob_fake:.1f}% {status_symbol}")
    
    print("\n[=] Summary:")
    for class_name in ['real', 'fake']:
        if results[class_name]['total'] > 0:
            acc = results[class_name]['correct'] / results[class_name]['total'] * 100
            print(f"  {class_name.upper()}: {results[class_name]['correct']}/{results[class_name]['total']} "
                  f"({acc:.1f}%)")

if __name__ == "__main__":
    test_model()