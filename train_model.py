import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import shutil
from pathlib import Path

class DeepfakeTrainer:
    def __init__(self):
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 30
        self.model = None
        self.history = None
        
    def create_model(self):
        """Create model using transfer learning with EfficientNetB0"""
        # Load pre-trained EfficientNetB0
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Create model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        print("Model created successfully!")
        return model
    
    def prepare_data(self, data_dir='dataset'):
        """Prepare data for training"""
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'validation')
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load training data
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['real', 'fake']  # real=0, fake=1
        )
        
        # Load validation data
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['real', 'fake']
        )
        
        self.train_generator = train_generator
        self.val_generator = val_generator
        
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        print(f"Classes: {train_generator.class_indices}")
        
        return train_generator, val_generator
    
    def train(self, epochs=30):
        """Train the model"""
        if self.model is None:
            self.create_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-7
            ),
            ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        # Train
        self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // self.batch_size,
            epochs=epochs,
            validation_data=self.val_generator,
            validation_steps=self.val_generator.samples // self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        os.makedirs('models', exist_ok=True)
        self.model.save('models/deepfake_detector.h5')
        
        # Save training history
        with open('models/training_history.pkl', 'wb') as f:
            pickle.dump(self.history.history, f)
        
        print("Model training completed and saved!")
        return self.history
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history found.")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        
        # Loss
        axes[1].plot(self.history.history['loss'], label='Training Loss')
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        
        # Precision/Recall
        if 'precision' in self.history.history:
            axes[2].plot(self.history.history['precision'], label='Precision')
            axes[2].plot(self.history.history['recall'], label='Recall')
            axes[2].set_title('Precision & Recall')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Score')
            axes[2].legend()
        
        plt.tight_layout()
        plt.savefig('models/training_history.png')
        plt.show()
    
    def evaluate_model(self, test_dir='dataset/test'):
        """Evaluate the model on test data"""
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['real', 'fake'],
            shuffle=False
        )
        
        results = self.model.evaluate(test_generator)
        
        print(f"Test Loss: {results[0]:.4f}")
        print(f"Test Accuracy: {results[1]:.4f}")
        print(f"Test Precision: {results[2]:.4f}")
        print(f"Test Recall: {results[3]:.4f}")
        
        return results
    
    def predict_image(self, image_path):
        """Predict if an image is real or fake"""
        if self.model is None:
            self.model = keras.models.load_model('models/deepfake_detector.h5')
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            return {'error': 'Could not load image'}
        
        # Detect face
        face = self.detect_face(img)
        if face is not None:
            img = face
        
        # Resize and normalize
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict
        prediction = self.model.predict(img, verbose=0)
        prob_fake = float(prediction[0][0] * 100)
        prob_real = 100 - prob_fake
        
        result = 'fake' if prob_fake > prob_real else 'real'
        confidence = max(prob_fake, prob_real)
        
        return {
            'result': result,
            'confidence': confidence,
            'probability_real': prob_real,
            'probability_fake': prob_fake
        }
    
    def detect_face(self, image):
        """Detect face in image using OpenCV"""
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
        
        if len(faces) > 0:
            # Get largest face
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            return image[y:y+h, x:x+w]
        
        return None
    
    def create_sample_dataset(self):
        """Create a sample dataset for testing (for demonstration)"""
        print("Creating sample dataset structure...")
        
        # Create directories
        for split in ['train', 'validation', 'test']:
            for cls in ['real', 'fake']:
                os.makedirs(f'dataset/{split}/{cls}', exist_ok=True)
        
        print("""
        ╔══════════════════════════════════════════════════════════════╗
        ║                    DATASET SETUP GUIDE                      ║
        ╚══════════════════════════════════════════════════════════════╝
        
        You need to add images to the following folders:
        
        1. REAL IMAGES (Place real face images):
           dataset/train/real/
           dataset/validation/real/
           dataset/test/real/
        
        2. FAKE/AI-GENERATED IMAGES (Place AI-generated face images):
           dataset/train/fake/
           dataset/validation/fake/
           dataset/test/fake/
        
        RECOMMENDED DATASETS:
        • Celeb-DF: https://github.com/yuezunli/Celeb-DF
        • FaceForensics++: https://github.com/ondyari/FaceForensics
        • 140k Real and Fake Faces: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
        
        For quick testing, you can use the "140k Real and Fake Faces" dataset from Kaggle.
        """)

# Main execution
if __name__ == "__main__":
    trainer = DeepfakeTrainer()
    
    # Check if model exists
    if os.path.exists('models/deepfake_detector.h5'):
        print("Loading existing model...")
        trainer.model = keras.models.load_model('models/deepfake_detector.h5')
    else:
        print("No existing model found. Creating new model...")
        trainer.create_model()
    
    # Check if dataset exists
    train_dir = 'dataset/train'
    if os.path.exists(train_dir) and len(os.listdir(os.path.join(train_dir, 'real'))) > 0:
        print("Training with existing dataset...")
        trainer.prepare_data()
        trainer.train(epochs=30)
        trainer.plot_training_history()
        trainer.evaluate_model()
    else:
        print("No dataset found. Please add images to the dataset folders.")
        trainer.create_sample_dataset()
        print("\nAfter adding images, run this script again to train the model.")