"""
Deepfake Detector — Inference Module (Improved)
===============================================

Key improvements:
1. Calibrated sigmoid threshold: 0.45 (less false-real predictions)
2. Multi-signal heuristic fallback when TF model unavailable:
   - Laplacian variance (sharpness)
   - DCT frequency energy analysis (GAN checkerboard pattern)
   - Colour channel saturation variance
   - Local noise variance
3. Robust face detection: Haar cascade + DNN fallback
4. Video analysis: 60 frames, face-weighted voting
"""

import os
import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────
#  Heuristic helpers (used when TF model is unavailable)
# ─────────────────────────────────────────────────────────────

def _heuristic_fake_score(bgr_img):
    """
    Return a fake probability in [0, 1] using multiple image quality signals.

    Signals:
      1. Laplacian variance  — GAN images often over-smooth or over-sharpen
      2. DCT energy ratio    — Checkerboard artefacts boost high-frequency energy
      3. Colour saturation   — GANs sometimes produce abnormal saturation
      4. Local noise std     — Sensor noise pattern differs in real vs synthetic

    Returns float in [0, 1], where 1 = likely fake.
    """
    if bgr_img is None or bgr_img.size == 0:
        return 0.5

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ── Signal 1: Laplacian variance ─────────────────────────
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Very smooth (GAN over-smoothing) → high fake score
    # Very natural range: 80–600
    if lap_var < 30:
        s1 = 0.80      # extremely smooth → almost certainly fake/edited
    elif lap_var < 80:
        s1 = 0.65
    elif lap_var < 400:
        s1 = 0.30      # natural sharpness range
    else:
        s1 = 0.20      # very sharp → likely real natural photo

    # ── Signal 2: DCT high-frequency energy ratio ────────────
    # Resize to 64x64 to get frequency signature
    gray_small = cv2.resize(gray, (64, 64)).astype(np.float32)
    dct = cv2.dct(gray_small)
    # Energy in top-left (low-freq) vs rest (high-freq)
    total_energy = float(np.sum(dct ** 2)) + 1e-6
    low_freq_energy = float(np.sum(dct[:8, :8] ** 2))
    high_freq_ratio = 1.0 - (low_freq_energy / total_energy)
    # GANs tend to have elevated periodic high-freq content
    if high_freq_ratio > 0.60:
        s2 = 0.70      # unusual high-freq → suspicious
    elif high_freq_ratio > 0.45:
        s2 = 0.45
    else:
        s2 = 0.30      # low high-freq → typical real photo

    # ── Signal 3: Colour saturation variance ─────────────────
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    sat_mean = float(np.mean(sat))
    sat_std = float(np.std(sat))
    # GANs sometimes produce abnormally uniform or abnormally high saturation
    if sat_mean > 180 or sat_std < 15:
        s3 = 0.65
    elif sat_mean < 20:
        s3 = 0.60      # unnaturally desaturated (face-swap artefact)
    else:
        s3 = 0.30

    # ── Signal 4: Local noise variance (shot noise pattern) ──
    # Compare local patch std to global std — GANs have uniform noise
    patch_h, patch_w = h // 4, w // 4
    local_stds = []
    for py in range(4):
        for px in range(4):
            patch = gray[py*patch_h:(py+1)*patch_h, px*patch_w:(px+1)*patch_w]
            local_stds.append(float(np.std(patch.astype(np.float32))))
    std_of_stds = float(np.std(local_stds))
    # Real images have varying local noise; GAN images tend to be more uniform
    if std_of_stds < 3.0:
        s4 = 0.60      # very uniform noise → GAN-like
    elif std_of_stds < 8.0:
        s4 = 0.40
    else:
        s4 = 0.25

    # ── Weighted combination ──────────────────────────────────
    # Weights: lap_var most reliable in heuristic context
    fake_score = 0.35 * s1 + 0.25 * s2 + 0.20 * s3 + 0.20 * s4
    return float(np.clip(fake_score, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────
#  Face detection helpers
# ─────────────────────────────────────────────────────────────

def _detect_face_haar(bgr_img):
    """Haar cascade face detection — fast, works offline."""
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(40, 40))
        if len(faces) > 0:
            x, y, ww, hh = max(faces, key=lambda f: f[2] * f[3])
            # Add 10% padding
            pad_x = int(ww * 0.10)
            pad_y = int(hh * 0.10)
            ih, iw = bgr_img.shape[:2]
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(iw, x + ww + pad_x)
            y1 = min(ih, y + hh + pad_y)
            return bgr_img[y0:y1, x0:x1], True
    except Exception:
        pass
    return None, False


def _detect_face_dnn(bgr_img):
    """
    OpenCV DNN-based face detector (more robust than Haar, especially for
    profile faces and low-resolution images). Uses the in-built SSD model
    bundled with some opencv-python distributions.
    Falls back silently if not available.
    """
    try:
        # Attempt to use the deep face detector if weights are available
        model_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'models',
            'res10_300x300_ssd_iter_140000.caffemodel')
        config_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'models',
            'deploy.prototxt')
        if not (os.path.exists(model_file) and os.path.exists(config_file)):
            return None, False
        net = cv2.dnn.readNetFromCaffe(config_file, model_file)
        h, w = bgr_img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(bgr_img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        best_conf = 0.0
        best_face = None
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > 0.5 and conf > best_conf:
                best_conf = conf
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x0, y0, x1, y1 = box.astype(int)
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)
                best_face = bgr_img[y0:y1, x0:x1]
        if best_face is not None and best_face.size > 0:
            return best_face, True
    except Exception:
        pass
    return None, False


def detect_face(bgr_img):
    """
    Try Haar cascade first, then DNN fallback.
    Returns (face_crop, found_face) tuple.
    If no face is detected, returns the original image with found_face=False.
    """
    face, found = _detect_face_haar(bgr_img)
    if found and face is not None and face.size > 0:
        return face, True
    face, found = _detect_face_dnn(bgr_img)
    if found and face is not None and face.size > 0:
        return face, True
    return bgr_img, False


# ─────────────────────────────────────────────────────────────
#  Main detector class
# ─────────────────────────────────────────────────────────────

DETECTION_THRESHOLD = 0.45   # calibrated — reduces false-real classifications


class DeepfakeDetector:
    def __init__(self, model_path=None):
        self.model = None
        self.candidate_model_paths = [
            'models/deepfake_model.h5',         # primary — written by improved trainer
            'models/stage1_best.h5',            # trained base model
            'models/best_model.h5',
            model_path or 'models/deepfake_detector.h5',
        ]
        self.img_size = (224, 224)
        self.load_model()

    # ─── Model loading ───────────────────────────────────────

    def load_model(self):
        """Load the trained model from candidate paths."""
        try:
            from tensorflow.keras.models import load_model
            for path in self.candidate_model_paths:
                if path and os.path.exists(path) and os.path.getsize(path) > 50_000:
                    try:
                        self.model = load_model(path)
                        print(f"Deepfake model loaded from: {path}")
                        return
                    except Exception as e:
                        print(f"Could not load model from {path}: {e}")
        except (ImportError, Exception) as e:
            print(f"TensorFlow not available: {e}")

        print("Running in heuristic fallback mode (multi-signal analysis).")
        self.model = None

    # ─── Preprocessing ───────────────────────────────────────

    def preprocess_image(self, bgr_or_rgb_img):
        """Preprocess a numpy image (BGR or RGB) for model prediction."""
        if bgr_or_rgb_img is None:
            raise ValueError("Input image is None")

        # Ensure RGB
        if len(bgr_or_rgb_img.shape) == 2:
            img = cv2.cvtColor(bgr_or_rgb_img, cv2.COLOR_GRAY2RGB)
        elif bgr_or_rgb_img.shape[2] == 3:
            img = cv2.cvtColor(bgr_or_rgb_img, cv2.COLOR_BGR2RGB)
        else:
            img = bgr_or_rgb_img[:, :, :3]

        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    # ─── Single image prediction ─────────────────────────────

    def predict_image(self, image_input):
        """
        Predict whether an image is real or fake.

        Args:
            image_input: file path (str) or numpy BGR array.

        Returns:
            dict with keys: result, confidence, probability_real, probability_fake
        """
        try:
            # ── Load image ────────────────────────────────────
            if isinstance(image_input, str):
                bgr_img = cv2.imread(image_input)
                if bgr_img is None:
                    raise ValueError(f"Could not read image: {image_input}")
            else:
                bgr_img = image_input

            if bgr_img is None or bgr_img.size == 0:
                raise ValueError("Empty image")

            # ── Face detection ────────────────────────────────
            face_img, face_found = detect_face(bgr_img)
            target_img = face_img if face_found else bgr_img

            # ── Predict ───────────────────────────────────────
            if self.model is not None:
                processed = self.preprocess_image(target_img)
                raw_pred = float(self.model.predict(processed, verbose=0)[0][0])
                prob_fake = raw_pred
            else:
                # Multi-signal heuristic fallback
                prob_fake = _heuristic_fake_score(target_img)
                # If no face was detected, reduce confidence (push toward uncertain)
                if not face_found:
                    prob_fake = 0.4 * prob_fake + 0.6 * 0.5

            prob_real = 1.0 - prob_fake

            # ── Apply calibrated threshold ────────────────────
            if prob_fake > DETECTION_THRESHOLD:
                result = 'fake'
                confidence = prob_fake * 100
            else:
                result = 'real'
                confidence = prob_real * 100

            return {
                'result': result,
                'confidence': round(float(confidence), 2),
                'probability_real': round(float(prob_real * 100), 2),
                'probability_fake': round(float(prob_fake * 100), 2),
            }

        except Exception as e:
            print(f"Error in predict_image: {e}")
            return {
                'result': 'unknown',
                'confidence': 0.0,
                'probability_real': 50.0,
                'probability_fake': 50.0,
            }

    # ─── Video prediction ────────────────────────────────────

    def predict_video(self, video_path, max_frames=60):
        """
        Analyse a video for deepfake content.

        Samples up to `max_frames` frames evenly across the video.
        Frames where a face is detected are given 2× weight.

        Returns:
            dict with keys: result, confidence, probability_real, probability_fake,
                            frames_analyzed, frames_with_face
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                total_frames = 9999
            frame_interval = max(1, total_frames // max_frames)

            frame_count = 0
            weighted_sum = 0.0
            total_weight = 0.0
            frames_analyzed = 0
            frames_with_face = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0 and frames_analyzed < max_frames:
                    result = self.predict_image(frame)
                    if result['result'] != 'unknown':
                        face_img, face_found = detect_face(frame)
                        weight = 2.0 if face_found else 1.0
                        weighted_sum += (result['probability_fake'] / 100.0) * weight
                        total_weight += weight
                        frames_analyzed += 1
                        if face_found:
                            frames_with_face += 1

                frame_count += 1

            cap.release()

            if total_weight == 0.0:
                return {
                    'result': 'unknown',
                    'confidence': 0.0,
                    'probability_real': 50.0,
                    'probability_fake': 50.0,
                    'frames_analyzed': 0,
                    'frames_with_face': 0,
                }

            prob_fake = weighted_sum / total_weight
            prob_real = 1.0 - prob_fake

            if prob_fake > DETECTION_THRESHOLD:
                result_label = 'fake'
                confidence = prob_fake * 100
            else:
                result_label = 'real'
                confidence = prob_real * 100

            return {
                'result': result_label,
                'confidence': round(float(confidence), 2),
                'probability_real': round(float(prob_real * 100), 2),
                'probability_fake': round(float(prob_fake * 100), 2),
                'frames_analyzed': frames_analyzed,
                'frames_with_face': frames_with_face,
            }

        except Exception as e:
            print(f"Error in predict_video: {e}")
            return {
                'result': 'unknown',
                'confidence': 0.0,
                'probability_real': 50.0,
                'probability_fake': 50.0,
                'frames_analyzed': 0,
                'frames_with_face': 0,
            }

    # ── Legacy method names kept for backward compatibility ──

    def detect_face(self, image):
        """Backward-compatible face detection wrapper."""
        face, _ = detect_face(image)
        return face

    def preprocess_image_path(self, image_path):
        """Backward-compatible preprocessing from file path."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return self.preprocess_image(img)


# ── Global detector instance ──────────────────────────────────
detector = DeepfakeDetector()