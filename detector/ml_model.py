import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import joblib

class DeepfakeDetector:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path or 'models/deepfake_detector.h5'
        self.best_model_path = 'models/best_model.h5'
        self.img_size = (224, 224)
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        # Try to load best model first
        if os.path.exists(self.best_model_path):
            try:
                self.model = load_model(self.best_model_path)
                print(f"Best model loaded from {self.best_model_path}")
                return
            except Exception as e:
                print(f"Error loading best model: {e}")
        
        # Fallback to regular model
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                print(f"Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.create_dummy_model()
        else:
            print("Model not found. Creating dummy model...")
            self.create_dummy_model()
    
    def create_dummy_model(self):
        """Create a dummy model for demonstration"""
        try:
            from tensorflow.keras import layers, models
            self.model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                layers.Flatten(),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(1, activation='sigmoid')
            ])
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            print("Dummy model created for demonstration.")
        except Exception as e:
            print(f"Error creating dummy model: {e}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
        else:
            img = image_path
        
        if img is None:
            raise ValueError("Could not load image")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, self.img_size)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # Expand dimensions
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def detect_face(self, image):
        """Detect face in image"""
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        # Get largest face
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        face = image[y:y+h, x:x+w]
        
        return face
    
    def predict_image(self, image_path):
        """Predict if image is real or fake"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not load image")
            
            # Detect face
            face = self.detect_face(img)
            
            if face is not None:
                img = face
            
            # Preprocess
            processed_img = self.preprocess_image(img)
            
            # Predict
            prediction = self.model.predict(processed_img, verbose=0)
            prob_fake = float(prediction[0][0] * 100)
            prob_real = 100 - prob_fake
            
            # Determine result
            if prob_real > prob_fake:
                result = 'real'
                confidence = prob_real
            else:
                result = 'fake'
                confidence = prob_fake
            
            return {
                'result': result,
                'confidence': confidence,
                'probability_real': prob_real,
                'probability_fake': prob_fake
            }
        
        except Exception as e:
            print(f"Error in prediction: {e}")
            return {
                'result': 'unknown',
                'confidence': 0.0,
                'probability_real': 50.0,
                'probability_fake': 50.0
            }
    
    def predict_video(self, video_path, max_frames=30):
        """Predict if video contains deepfake"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError("Could not open video")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = max(1, total_frames // max_frames)
            frame_count = 0
            predictions = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0 and len(predictions) < max_frames:
                    # Detect face
                    face = self.detect_face(frame)
                    if face is not None:
                        processed = self.preprocess_image(face)
                        pred = self.model.predict(processed, verbose=0)
                        predictions.append(pred[0][0])
                
                frame_count += 1
            
            cap.release()
            
            if len(predictions) == 0:
                return {
                    'result': 'unknown',
                    'confidence': 0.0,
                    'probability_real': 50.0,
                    'probability_fake': 50.0
                }
            
            # Aggregate predictions
            avg_prediction = np.mean(predictions)
            prob_fake = float(avg_prediction * 100)
            prob_real = 100 - prob_fake
            
            if prob_real > prob_fake:
                result = 'real'
                confidence = prob_real
            else:
                result = 'fake'
                confidence = prob_fake
            
            return {
                'result': result,
                'confidence': confidence,
                'probability_real': prob_real,
                'probability_fake': prob_fake
            }
        
        except Exception as e:
            print(f"Error in video prediction: {e}")
            return {
                'result': 'unknown',
                'confidence': 0.0,
                'probability_real': 50.0,
                'probability_fake': 50.0
            }

# Create global detector instance
detector = DeepfakeDetector()