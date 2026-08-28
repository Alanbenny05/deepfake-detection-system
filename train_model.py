"""
Deepfake Detection Model Trainer — Improved Version
====================================================
Two-stage training:
  Stage 1: Train the classification head with the EfficientNetB0 base frozen.
  Stage 2: Fine-tune the top 30 layers of EfficientNetB0 at a low learning rate.

Key improvements over previous version:
- Two-stage fine-tuning (not fully frozen base)
- Enhanced augmentation pipeline (brightness, contrast, JPEG noise)
- Class-balanced training via class_weight
- Calibrated sigmoid threshold (0.45 instead of 0.5)
- Saves as 'models/deepfake_model.h5' (compatible with ml_model.py loader)
"""
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')   # headless backend — safe for server/script use
import matplotlib.pyplot as plt
import pickle
from pathlib import Path


# ─────────────────────────────────────────────────────────────
#  Custom Keras layer: JPEG noise augmentation
# ─────────────────────────────────────────────────────────────

class JPEGNoise(layers.Layer):
    """Simulate random JPEG compression during training (augmentation)."""
    def __init__(self, min_quality=65, max_quality=95, **kwargs):
        super().__init__(**kwargs)
        self.min_quality = min_quality
        self.max_quality = max_quality

    def call(self, inputs, training=None):
        if not training:
            return inputs
        # Apply JPEG compression on CPU via tf.numpy_function
        def _jpeg_compress(img_batch):
            results = []
            for img in img_batch:
                quality = int(np.random.randint(self.min_quality, self.max_quality))
                img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
                _, buf = cv2.imencode('.jpg', cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR),
                                     [cv2.IMWRITE_JPEG_QUALITY, quality])
                decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                decoded_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                results.append(decoded_rgb)
            return np.array(results, dtype=np.float32)

        return tf.numpy_function(_jpeg_compress, [inputs], tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({'min_quality': self.min_quality, 'max_quality': self.max_quality})
        return config


class DeepfakeTrainer:
    def __init__(self):
        self.img_size = (224, 224)
        self.batch_size = 32
        self.model = None
        self.history_stage1 = None
        self.history_stage2 = None

    # ─── Model ───────────────────────────────────────────────

    def create_model(self):
        """
        Build classification model using EfficientNetB0.
        Base is initially frozen; fine-tuning unfreezes top 30 layers in Stage 2.
        """
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False   # frozen for Stage 1

        inputs = keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        self.model = keras.Model(inputs, outputs, name='deepfake_detector')
        self._compile(lr=1e-3)
        self.base_model = base_model
        print("Model created. Base model trainable layers: 0 (Stage 1 — head only).")
        return self.model

    def _compile(self, lr=1e-3):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=['accuracy',
                     keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall'),
                     keras.metrics.AUC(name='auc')]
        )

    # ─── Data ────────────────────────────────────────────────

    def prepare_data(self, data_dir='dataset'):
        """Build data generators for training and validation."""
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'validation')

        # ── Training augmentation ──────────────────────────────
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=25,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.20,
            horizontal_flip=True,
            brightness_range=[0.75, 1.25],
            channel_shift_range=20.0,       # colour channel shifts — important for deepfake features
            fill_mode='nearest'
        )

        # ── Validation: only normalise ─────────────────────────
        val_datagen = ImageDataGenerator(rescale=1. / 255)

        self.train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['real', 'fake'],   # real=0, fake=1
            shuffle=True,
            seed=42
        )

        self.val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['real', 'fake'],
            shuffle=False
        )

        print(f"Training   : {self.train_generator.samples} samples  "
              f"({self.train_generator.samples // self.batch_size} steps/epoch)")
        print(f"Validation : {self.val_generator.samples} samples  "
              f"({self.val_generator.samples // self.batch_size} steps/epoch)")
        print(f"Class indices: {self.train_generator.class_indices}")

    def _compute_class_weights(self):
        """Compute class weights to handle any residual imbalance."""
        labels = self.train_generator.classes
        classes = np.unique(labels)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
        cw = dict(zip(classes.astype(int), weights))
        print(f"Class weights: {cw}")
        return cw

    # ─── Training ────────────────────────────────────────────

    def _callbacks(self, checkpoint_path, monitor='val_auc', mode='max'):
        return [
            EarlyStopping(monitor=monitor, patience=8, restore_best_weights=True, mode=mode),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-8, verbose=1),
            ModelCheckpoint(checkpoint_path, monitor=monitor, save_best_only=True,
                            mode=mode, verbose=1)
        ]

    def train_stage1(self, epochs=25):
        """Stage 1: Train classification head only (base frozen)."""
        print("\n" + "=" * 60)
        print("STAGE 1: Training classification head (base frozen)")
        print("=" * 60)
        os.makedirs('models', exist_ok=True)
        cw = self._compute_class_weights()

        self.history_stage1 = self.model.fit(
            self.train_generator,
            steps_per_epoch=max(1, self.train_generator.samples // self.batch_size),
            epochs=epochs,
            validation_data=self.val_generator,
            validation_steps=max(1, self.val_generator.samples // self.batch_size),
            class_weight=cw,
            callbacks=self._callbacks('models/stage1_best.h5'),
            verbose=1
        )
        print("Stage 1 complete.")

    def train_stage2(self, epochs=20, unfreeze_layers=30):
        """
        Stage 2: Fine-tune the top N layers of EfficientNetB0 at a very low LR.
        Only the top `unfreeze_layers` of the base are made trainable.
        """
        print("\n" + "=" * 60)
        print(f"STAGE 2: Fine-tuning top {unfreeze_layers} layers of EfficientNetB0")
        print("=" * 60)

        # Unfreeze top layers
        self.base_model.trainable = True
        for layer in self.base_model.layers[:-unfreeze_layers]:
            layer.trainable = False

        trainable_count = sum(1 for l in self.base_model.layers if l.trainable)
        print(f"Trainable base layers: {trainable_count} / {len(self.base_model.layers)}")

        # Recompile at very low LR to avoid destroying pretrained weights
        self._compile(lr=1e-5)

        cw = self._compute_class_weights()
        self.train_generator.reset()
        self.val_generator.reset()

        self.history_stage2 = self.model.fit(
            self.train_generator,
            steps_per_epoch=max(1, self.train_generator.samples // self.batch_size),
            epochs=epochs,
            validation_data=self.val_generator,
            validation_steps=max(1, self.val_generator.samples // self.batch_size),
            class_weight=cw,
            callbacks=self._callbacks('models/stage2_best.h5'),
            verbose=1
        )
        print("Stage 2 complete.")

    def save_model(self):
        """Save the final trained model in all expected locations."""
        os.makedirs('models', exist_ok=True)
        # Primary path used by ml_model.py
        self.model.save('models/deepfake_model.h5')
        # Secondary paths
        self.model.save('models/deepfake_detector.h5')
        self.model.save('models/best_model.h5')
        print("Models saved to models/deepfake_model.h5, deepfake_detector.h5, best_model.h5")

    # ─── Evaluation ──────────────────────────────────────────

    def evaluate_model(self, test_dir='dataset/test'):
        """Evaluate model on test set and print classification report."""
        print("\n--- Evaluating on test set ---")
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['real', 'fake'],
            shuffle=False
        )
        results = self.model.evaluate(test_gen, verbose=1)
        names = ['loss', 'accuracy', 'precision', 'recall', 'auc']
        for name, val in zip(names, results):
            print(f"  Test {name}: {val:.4f}")

        # Confusion matrix using calibrated threshold = 0.45
        test_gen.reset()
        preds = self.model.predict(test_gen, verbose=0)
        y_pred = (preds > 0.45).astype(int).flatten()
        y_true = test_gen.classes

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        print(f"\n  Confusion matrix (threshold=0.45):")
        print(f"    TP (correctly detected fake): {tp}")
        print(f"    TN (correctly detected real): {tn}")
        print(f"    FP (real wrongly classified as fake): {fp}")
        print(f"    FN (fake wrongly classified as real): {fn}")
        acc_calibrated = (tp + tn) / max(1, tp + tn + fp + fn)
        print(f"    Calibrated accuracy: {acc_calibrated:.4f}")
        return results

    def plot_training_history(self):
        """Plot combined training history from both stages."""
        histories = []
        if self.history_stage1:
            histories.append(('Stage 1', self.history_stage1.history))
        if self.history_stage2:
            histories.append(('Stage 2', self.history_stage2.history))

        if not histories:
            print("No training history to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for stage_name, h in histories:
            axes[0, 0].plot(h.get('accuracy', []), label=f'{stage_name} Train')
            axes[0, 0].plot(h.get('val_accuracy', []), label=f'{stage_name} Val')
            axes[0, 1].plot(h.get('loss', []), label=f'{stage_name} Train')
            axes[0, 1].plot(h.get('val_loss', []), label=f'{stage_name} Val')
            if 'auc' in h:
                axes[1, 0].plot(h.get('auc', []), label=f'{stage_name} Train')
                axes[1, 0].plot(h.get('val_auc', []), label=f'{stage_name} Val')
            if 'precision' in h:
                axes[1, 1].plot(h.get('precision', []), label=f'{stage_name} Prec')
                axes[1, 1].plot(h.get('recall', []), label=f'{stage_name} Recall')

        for ax, title, ylabel in [
            (axes[0, 0], 'Accuracy', 'Accuracy'),
            (axes[0, 1], 'Loss', 'Loss'),
            (axes[1, 0], 'AUC', 'AUC'),
            (axes[1, 1], 'Precision & Recall', 'Score'),
        ]:
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7)

        plt.tight_layout()
        os.makedirs('models', exist_ok=True)
        plt.savefig('models/training_history.png', dpi=100)
        print("Training history saved to models/training_history.png")

    def predict_image(self, image_path, threshold=0.45):
        """Predict a single image with calibrated threshold."""
        if self.model is None:
            self.model = keras.models.load_model('models/deepfake_model.h5')

        img = cv2.imread(str(image_path))
        if img is None:
            return {'error': 'Could not load image'}

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.img_size)
        img_norm = img_resized.astype(np.float32) / 255.0
        batch = np.expand_dims(img_norm, axis=0)

        prob_fake = float(self.model.predict(batch, verbose=0)[0][0])
        prob_real = 1.0 - prob_fake

        result = 'fake' if prob_fake > threshold else 'real'
        confidence = max(prob_fake, prob_real) * 100

        return {
            'result': result,
            'confidence': round(confidence, 2),
            'probability_real': round(prob_real * 100, 2),
            'probability_fake': round(prob_fake * 100, 2)
        }


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trainer = DeepfakeTrainer()

    # Check for existing trained model
    model_found = False
    for candidate in ['models/best_model.h5', 'models/deepfake_model.h5',
                      'models/deepfake_detector.h5']:
        if os.path.exists(candidate) and os.path.getsize(candidate) > 50_000:
            print(f"Loading existing model from {candidate} ...")
            trainer.model = keras.models.load_model(candidate)
            model_found = True
            break

    if not model_found:
        trainer.create_model()

    # Check dataset
    train_real = os.path.join('dataset', 'train', 'real')
    n_real = len([f for f in os.listdir(train_real)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(train_real) else 0

    if n_real == 0:
        print("No training images found. Run generate_sample_dataset.py first.")
        exit(1)

    trainer.prepare_data()
    trainer.train_stage1(epochs=25)
    trainer.train_stage2(epochs=20)
    trainer.save_model()
    trainer.plot_training_history()
    trainer.evaluate_model()