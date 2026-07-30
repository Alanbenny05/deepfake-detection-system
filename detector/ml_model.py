import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import joblib

class DeepfakeDetector:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path or 'models/deepfake_detector.h5'
        self.image_size = (128, 128)
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                print(f"Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.create_model()
        else:
            print("Model not found. Creating new model...")
            self.create_model()
    
    def create_model(self):
        """Create a new CNN model"""
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(2, activation='softmax')  # Real vs Fake
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("New model created")
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        # Read image
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
        else:
            img = image_path
        
        if img is None:
            raise ValueError("Could not load image")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, self.image_size)
        
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
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        # Get the largest face
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        
        # Extract face
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
            
            if face is None:
                # If no face detected, use whole image
                face = img
            
            # Preprocess
            processed_img = self.preprocess_image(face)
            
            # Predict
            prediction = self.model.predict(processed_img, verbose=0)
            
            # Get probabilities
            prob_real = float(prediction[0][0] * 100)
            prob_fake = float(prediction[0][1] * 100)
            
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
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Could not open video")
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract frames at intervals
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
                    # Predict
                    processed = self.preprocess_image(face)
                    pred = self.model.predict(processed, verbose=0)
                    predictions.append(pred[0])
            
            frame_count += 1
        
        cap.release()
        
        if len(predictions) == 0:
            # If no faces detected, use whole video
            return {'result': 'unknown', 'confidence': 0.0}
        
        # Aggregate predictions
        avg_prediction = np.mean(predictions, axis=0)
        prob_real = float(avg_prediction[0] * 100)
        prob_fake = float(avg_prediction[1] * 100)
        
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
    
    def train_model(self, dataset_path, epochs=10):
        """Train the model with dataset"""
        # Data augmentation
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        # Load training data
        train_generator = datagen.flow_from_directory(
            dataset_path,
            target_size=self.image_size,
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        
        validation_generator = datagen.flow_from_directory(
            dataset_path,
            target_size=self.image_size,
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )
        
        # Train model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            verbose=1
        )
        
        # Save model
        os.makedirs('models', exist_ok=True)
        self.model.save(self.model_path)
        
        return history
    
    def save_model(self):
        """Save the model"""
        os.makedirs('models', exist_ok=True)
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def load_sample_dataset(self):
        """Load and prepare sample dataset for training"""
        # This is a placeholder - you need to download actual dataset
        # For production, use Celeb-DF, FaceForensics++, or DeepFakeDetection dataset
        print("Please download a proper dataset for training")
        print("Recommended datasets:")
        print("1. Celeb-DF (https://github.com/yuezunli/Celeb-DF)")
        print("2. FaceForensics++ (https://github.com/ondyari/FaceForensics)")
        print("3. DeepFakeDetection (https://ai.googleblog.com/2019/09/contributing-data-to-deepfake.html)")

# Initialize detector
detector = DeepfakeDetector()

# For demonstration, create a mock model if not exists
if not os.path.exists('models/deepfake_detector.h5'):
    print("Creating sample model for demonstration...")
    # Create dummy model with random weights
    detector.create_model()
    detector.save_model()