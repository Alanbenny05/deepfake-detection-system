"""
Face Detector Module
====================
Dedicated face detection service for the deepfake detection pipeline.

Provides:
  - Haar cascade detection (fast, offline)
  - OpenCV DNN detection fallback (more robust)
  - Face quality scoring (blur, size adequacy)
  - Structured result dict for use by other detectors
"""

import os
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

FACE_PADDING = 0.12          # fractional padding around detected face region
MIN_FACE_SIZE = (40, 40)     # minimum pixel size accepted as a face
DNN_CONFIDENCE_THRESHOLD = 0.50

# Path to optional DNN face detector model files (bundled in some distributions)
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
_DNN_WEIGHTS = os.path.join(_MODEL_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')
_DNN_CONFIG  = os.path.join(_MODEL_DIR, 'deploy.prototxt')


# ─────────────────────────────────────────────────────────────────────────────
#  Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _add_padding(x, y, w, h, img_h, img_w, pad_frac=FACE_PADDING):
    """Expand a face bounding box by pad_frac on each side, clamped to image."""
    pad_x = int(w * pad_frac)
    pad_y = int(h * pad_frac)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(img_w, x + w + pad_x)
    y1 = min(img_h, y + h + pad_y)
    return x0, y0, x1, y1


def _face_quality_score(face_crop):
    """
    Score the quality of a face crop in [0, 1].
    
    Considers:
      - Blur level (Laplacian variance) — very blurry faces hurt CNN accuracy
      - Size adequacy — very small crops lose detail
    
    Returns float in [0, 1] where 1 = excellent quality.
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0

    h, w = face_crop.shape[:2]

    # Size score
    size_score = min(1.0, (h * w) / (112 * 112))  # 112×112 is adequate

    # Sharpness score
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(1.0, lap_var / 200.0)  # 200+ = sharp

    quality = 0.5 * size_score + 0.5 * sharpness_score
    return float(np.clip(quality, 0.0, 1.0))


def _detect_haar(bgr_img):
    """Haar cascade face detection — fast, fully offline, trying multiple cascades for robustness."""
    cascades = [
        'haarcascade_frontalface_default.xml',
        'haarcascade_frontalface_alt2.xml',
        'haarcascade_frontalface_alt.xml'
    ]
    try:
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        for cascade_name in cascades:
            cascade_path = cv2.data.haarcascades + cascade_name
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if face_cascade.empty():
                continue
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=MIN_FACE_SIZE
            )
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest face
                ih, iw = bgr_img.shape[:2]
                x0, y0, x1, y1 = _add_padding(x, y, w, h, ih, iw)
                crop = bgr_img[y0:y1, x0:x1]
                if crop.size > 0:
                    return crop, (x0, y0, x1, y1), f'haar_{cascade_name.split("_")[1]}'
    except Exception as exc:
        logger.debug("Haar face detection failed: %s", exc)
    return None, None, None


def _detect_dnn(bgr_img):
    """OpenCV DNN face detector — more robust for profile/occluded faces."""
    if not (os.path.exists(_DNN_WEIGHTS) and os.path.exists(_DNN_CONFIG)):
        return None, None, None
    try:
        net = cv2.dnn.readNetFromCaffe(_DNN_CONFIG, _DNN_WEIGHTS)
        h, w = bgr_img.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(bgr_img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()

        best_conf = 0.0
        best_box = None
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf > DNN_CONFIDENCE_THRESHOLD and conf > best_conf:
                best_conf = conf
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                bx0, by0, bx1, by1 = box.astype(int)
                bx0, by0 = max(0, bx0), max(0, by0)
                bx1, by1 = min(w, bx1), min(h, by1)
                best_box = (bx0, by0, bx1, by1)

        if best_box is not None:
            bx0, by0, bx1, by1 = best_box
            crop = bgr_img[by0:by1, bx0:bx1]
            if crop.size > 0:
                return crop, best_box, 'dnn'
    except Exception as exc:
        logger.debug("DNN face detection failed: %s", exc)
    return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def detect_face(bgr_img):
    """
    Detect the primary face in a BGR image.

    Tries Haar cascade first (fast); falls back to DNN if available.

    Returns:
        dict with keys:
            found       (bool)
            crop        (np.ndarray BGR, or None)
            bbox        (tuple x0,y0,x1,y1, or None)
            method      ('haar' | 'dnn' | None)
            quality     (float 0–1; 0 if no face)
    """
    if bgr_img is None or bgr_img.size == 0:
        return _no_face_result("Empty image supplied")

    # Try Haar first
    crop, bbox, method = _detect_haar(bgr_img)
    if crop is None:
        crop, bbox, method = _detect_dnn(bgr_img)

    if crop is not None and crop.size > 0:
        quality = _face_quality_score(crop)
        logger.debug("Face detected via %s, quality=%.2f", method, quality)
        return {
            'found': True,
            'crop': crop,
            'bbox': bbox,
            'method': method,
            'quality': quality,
        }

    logger.debug("No face detected in image")
    return _no_face_result("No face detected")


def _no_face_result(reason=""):
    return {
        'found': False,
        'crop': None,
        'bbox': None,
        'method': None,
        'quality': 0.0,
        'reason': reason,
    }


# Backward-compat wrapper for ml_model.py usage
def detect_face_compat(bgr_img):
    """
    Backward-compatible wrapper that returns (face_crop, found_face)
    matching the old ml_model.py detect_face() signature.
    """
    result = detect_face(bgr_img)
    if result['found'] and result['crop'] is not None:
        return result['crop'], True
    return bgr_img, False
