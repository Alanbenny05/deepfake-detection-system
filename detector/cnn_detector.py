"""
CNN Detector Module
===================
Thin wrapper around the existing DeepfakeDetector class (ml_model.py).

Provides:
  - Structured result dict with status field
  - Graceful error handling — never crashes calling code
  - Backward compatibility with existing ml_model.py interface
"""

import logging
import os
import pickle
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CNNDetector:
    """
    Wrapper around the existing DeepfakeDetector that returns structured dicts
    and never raises exceptions to callers.
    """

    def __init__(self):
        self._detector = None
        self._classifier = None
        self._load_detector()
        self._load_classifier()

    def _load_detector(self):
        """Load the global detector from ml_model.py."""
        try:
            from .ml_model import detector as global_detector
            self._detector = global_detector
            model_loaded = self._detector.model is not None
            mode = "CNN model" if model_loaded else "heuristic fallback"
            logger.info("CNNDetector initialised - running in %s mode", mode)
        except Exception as exc:
            logger.error("CNNDetector: could not load ml_model detector: %s", exc)
            self._detector = None

    def _load_classifier(self):
        """Load the custom scikit-learn classifier pipeline if available."""
        pkl_path = 'models/deepfake_classifier.pkl'
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as f:
                    self._classifier = pickle.load(f)
                logger.info("Custom user classifier loaded successfully from %s", pkl_path)
            except Exception as exc:
                logger.error("Could not load custom classifier from %s: %s", pkl_path, exc)

    # ──────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────

    @property
    def model_available(self):
        """True if either the underlying TF model or trained classifier pipeline is loaded."""
        tf_loaded = self._detector is not None and self._detector.model is not None
        classifier_loaded = self._classifier is not None
        return tf_loaded or classifier_loaded

    @property
    def mode(self):
        """'cnn' | 'heuristic' | 'unavailable'"""
        if self._detector is None and self._classifier is None:
            return 'unavailable'
        if (self._detector is not None and self._detector.model is not None) or (self._classifier is not None):
            return 'cnn'
        return 'heuristic'

    def predict_image(self, image_input, face_crop=None):
        """
        Predict whether a single image (or face crop) is real or AI-generated.

        Args:
            image_input: str (file path) or np.ndarray (BGR)
            face_crop:   Optional pre-cropped face BGR array. If provided,
                         this is used for CNN inference instead of running
                         face detection again.

        Returns dict:
            {
                'status':        'ok' | 'error' | 'unavailable',
                'cnn_score':     float [0,1]  (probability of fake),
                'mode':          'cnn' | 'heuristic' | 'unavailable',
                'face_used':     bool,
                'error':         str or None,
            }
        """
        if self._detector is None:
            return self._unavailable_result("CNN detector module not loaded")

        t0 = time.monotonic()
        try:
            # 1. Base prediction
            face_used = False
            if face_crop is not None and face_crop.size > 0:
                base_score = self._predict_array_base(face_crop)
                face_used = True
            else:
                raw = self._detector.predict_image(image_input)
                if raw.get('result') == 'unknown':
                    base_score = 0.5
                else:
                    base_score = float(np.clip(raw.get('probability_fake', 50.0) / 100.0, 0.0, 1.0))

            # 2. Extract classifier features from full image if possible
            classifier_score = None
            if self._classifier is not None:
                full_bgr = None
                if isinstance(image_input, str):
                    full_bgr = cv2.imread(image_input)
                elif isinstance(image_input, np.ndarray):
                    full_bgr = image_input

                if full_bgr is not None and full_bgr.size > 0:
                    try:
                        feats = self._extract_classifier_features(full_bgr).reshape(1, -1)
                        probs = self._classifier.predict_proba(feats)[0]
                        classifier_score = float(probs[1]) # probability of fake
                    except Exception as exc:
                        logger.warning("Classifier prediction on full image failed: %s", exc)

            # 3. Blend predictions
            cnn_score = base_score
            if classifier_score is not None:
                cnn_score = float(np.clip(0.85 * classifier_score + 0.15 * base_score, 0.0, 1.0))
                logger.info("Blended image prediction: cnn_score=%.4f (classifier=%.4f, base_cnn=%.4f)",
                            cnn_score, classifier_score, base_score)

            elapsed = time.monotonic() - t0
            logger.debug(
                "CNNDetector.predict_image complete: cnn_score=%.3f mode=%s face_used=%s in %.2fs",
                cnn_score, self.mode, face_used, elapsed
            )

            return {
                'status':    'ok',
                'cnn_score': cnn_score,
                'mode':      self.mode,
                'face_used': face_used,
                'error':     None,
            }

        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.warning("CNNDetector.predict_image error (%.2fs): %s", elapsed, exc)
            return self._error_result(str(exc))

    def predict_array(self, bgr_array):
        """
        Convenience: predict directly from a numpy BGR array.
        No face detection is performed — pass the relevant crop.
        """
        if self._detector is None:
            return self._unavailable_result("CNN detector not loaded")
        try:
            return self._predict_array(bgr_array)
        except Exception as exc:
            logger.warning("CNNDetector.predict_array error: %s", exc)
            return self._error_result(str(exc))

    # ──────────────────────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────────────────────

    def _extract_classifier_features(self, img):
        """Extract identical 22-element feature vectors as train_user_images.py."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

        features = []

        # 1. Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.append(laplacian.var())
        features.append(np.mean(np.abs(laplacian)))
        features.append(np.max(np.abs(laplacian)))

        # 2. Color stats
        for c in range(3):
            features.append(np.mean(img_resized[:, :, c]))
            features.append(np.std(img_resized[:, :, c]))

        hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
        for c in range(3):
            features.append(np.mean(hsv[:, :, c]))
            features.append(np.std(hsv[:, :, c]))

        # 3. FFT
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)

        h, w = gray.shape
        cy, cx = h // 2, w // 2
        r_low = 30
        low_freq_mask = np.zeros_like(gray, dtype=bool)
        y, x = np.ogrid[:h, :w]
        low_freq_mask = (x - cx)**2 + (y - cy)**2 <= r_low**2

        low_energy = np.mean(magnitude_spectrum[low_freq_mask])
        high_energy = np.mean(magnitude_spectrum[~low_freq_mask])
        ratio = high_energy / (low_energy + 1e-6)
        features.extend([low_energy, high_energy, ratio])

        # 4. Sobel
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        features.append(np.mean(grad_mag))
        features.append(np.std(grad_mag))

        # 5. Noise residual
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_res = gray.astype(np.float32) - blurred.astype(np.float32)
        features.append(np.var(noise_res))
        features.append(np.mean(np.abs(noise_res)))

        return np.array(features, dtype=np.float32)

    def _predict_array_base(self, bgr_array):
        """Internal: run raw model or heuristic inference on a BGR numpy array."""
        if self._detector.model is not None:
            # TF CNN path
            processed = self._detector.preprocess_image(bgr_array)
            raw_pred = float(self._detector.model.predict(processed, verbose=0)[0][0])
            return float(np.clip(raw_pred, 0.0, 1.0))
        else:
            # Heuristic fallback (existing multi-signal logic)
            from .ml_model import _heuristic_fake_score
            return float(np.clip(_heuristic_fake_score(bgr_array), 0.0, 1.0))

    def _predict_array(self, bgr_array):
        """Internal: run inference on a BGR numpy array."""
        # 1. Base model prediction
        base_score = self._predict_array_base(bgr_array)

        # 2. Blend with custom user classifier if available (fallback to face crop if full image not provided)
        cnn_score = base_score
        if self._classifier is not None:
            try:
                feats = self._extract_classifier_features(bgr_array).reshape(1, -1)
                probs = self._classifier.predict_proba(feats)[0]
                classifier_score = float(probs[1]) # probability of fake
                # Blend: 85% classifier, 15% CNN base
                cnn_score = float(np.clip(0.85 * classifier_score + 0.15 * base_score, 0.0, 1.0))
                logger.debug("Blended custom classifier: cnn_score=%.4f (classifier=%.4f)", cnn_score, classifier_score)
            except Exception as exc:
                logger.warning("Failed custom classifier prediction: %s", exc)

        return {
            'status':    'ok',
            'cnn_score': cnn_score,
            'mode':      self.mode,
            'face_used': True,
            'error':     None,
        }

    @staticmethod
    def _unavailable_result(reason):
        return {
            'status':    'unavailable',
            'cnn_score': 0.5,
            'mode':      'unavailable',
            'face_used': False,
            'error':     reason,
        }

    @staticmethod
    def _error_result(reason):
        return {
            'status':    'error',
            'cnn_score': 0.5,
            'mode':      'error',
            'face_used': False,
            'error':     reason,
        }


# Module-level singleton
cnn_detector = CNNDetector()
