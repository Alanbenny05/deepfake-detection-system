"""
Image Detector Service
======================
Orchestrates the full image deepfake detection pipeline:

  1. Validate and load image
  2. Face detection
  3. CNN inference (on face crop if available)
  4. Forensic analysis
  5. Provenance check
  6. Ensemble decision
"""

import logging
import os
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Allowed image extensions
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}
MAX_IMAGE_DIMENSION = 8192   # pixels


def detect_image(file_path, media_type='image'):
    """
    Full image deepfake detection pipeline.

    Args:
        file_path:   str — absolute path to image file
        media_type:  str — 'image' (always for this function)

    Returns dict:
        {
            'status':            'ok' | 'error',
            'cnn_score':         float or None,
            'forensic_score':    float or None,
            'provenance_status': str,
            'provenance_markers': list[str],
            'final_score':       float,
            'classification':    str,
            'label':             str,
            'confidence':        float,
            'face_found':        bool,
            'face_quality':      float,
            'cnn_mode':          str,
            'explanation':       str,
            'signal_breakdown':  dict,
            'processing_time':   float,
            'error':             str or None,
        }
    """
    t_start = time.monotonic()

    # ── Validate ──────────────────────────────────────────────────────────────
    validation_error = _validate_image_path(file_path)
    if validation_error:
        return _error_result(validation_error, time.monotonic() - t_start)

    logger.info("Image detection started: %s", os.path.basename(file_path))

    # ── Load image ────────────────────────────────────────────────────────────
    bgr_img = cv2.imread(str(file_path))
    if bgr_img is None or bgr_img.size == 0:
        return _error_result("Could not read image file. It may be corrupt or unsupported.", time.monotonic() - t_start)

    logger.debug("Image loaded: %dx%d", bgr_img.shape[1], bgr_img.shape[0])

    # ── Face detection ────────────────────────────────────────────────────────
    try:
        from .face_detector import detect_face
        face_result = detect_face(bgr_img)
    except Exception as exc:
        logger.warning("Face detection error: %s", exc)
        face_result = {'found': False, 'crop': None, 'quality': 0.0}

    face_found   = face_result.get('found', False)
    face_crop    = face_result.get('crop')
    face_quality = face_result.get('quality', 0.0)
    logger.info("Face detection: found=%s quality=%.2f", face_found, face_quality)

    # ── CNN inference ─────────────────────────────────────────────────────────
    try:
        from .cnn_detector import cnn_detector
        cnn_result = cnn_detector.predict_image(bgr_img, face_crop if face_found else None)
        cnn_result['face_used'] = face_found
    except Exception as exc:
        logger.warning("CNN inference error: %s", exc)
        cnn_result = {'status': 'error', 'cnn_score': 0.5, 'mode': 'error', 'face_used': False, 'error': str(exc)}

    logger.info("CNN result: status=%s score=%.3f mode=%s",
                cnn_result.get('status'), cnn_result.get('cnn_score', 0.5), cnn_result.get('mode'))

    # ── Forensic analysis ─────────────────────────────────────────────────────
    # Analyse face crop if available (more focused); fall back to full image
    analysis_target = face_crop if (face_found and face_crop is not None) else bgr_img
    try:
        from . import forensic_detector
        forensic_result = forensic_detector.analyse(analysis_target)
    except Exception as exc:
        logger.warning("Forensic analysis error: %s", exc)
        forensic_result = {'status': 'error', 'forensic_score': 0.40, 'flags': [], 'error': str(exc)}

    logger.info("Forensic result: status=%s score=%.3f flags=%s",
                forensic_result.get('status'), forensic_result.get('forensic_score', 0.40),
                forensic_result.get('flags', []))

    # ── Provenance check ──────────────────────────────────────────────────────
    try:
        from .provenance.provenance_service import check_provenance
        provenance_result = check_provenance(file_path, media_type='image')
    except Exception as exc:
        logger.warning("Provenance check error: %s", exc)
        provenance_result = {'status': 'error', 'markers': [], 'message': str(exc)}

    logger.info("Provenance result: status=%s", provenance_result.get('status'))

    # ── Ensemble decision ─────────────────────────────────────────────────────
    try:
        from .ensemble.decision_engine import decide
        decision = decide(
            cnn_result=cnn_result,
            forensic_result=forensic_result,
            provenance_result=provenance_result,
            video_result=None,
            face_found=face_found,
            cnn_mode=cnn_result.get('mode', 'heuristic'),
        )
    except Exception as exc:
        logger.error("Ensemble decision error: %s", exc)
        decision = _fallback_decision(cnn_result, forensic_result)

    processing_time = time.monotonic() - t_start
    logger.info(
        "Image detection complete: %s (%.1f%% confidence) in %.2fs",
        decision.get('classification'), decision.get('confidence'), processing_time
    )

    return {
        'status':             'ok',
        'cnn_score':          cnn_result.get('cnn_score'),
        'cnn_mode':           cnn_result.get('mode', 'unknown'),
        'forensic_score':     forensic_result.get('forensic_score'),
        'forensic_flags':     forensic_result.get('flags', []),
        'provenance_status':  provenance_result.get('status', 'unavailable'),
        'provenance_markers': provenance_result.get('markers', []),
        'provenance_message': provenance_result.get('message', ''),
        'final_score':        decision.get('final_score', 0.5),
        'classification':     decision.get('classification', 'UNCERTAIN'),
        'label':              decision.get('label', 'Uncertain'),
        'confidence':         decision.get('confidence', 35.0),
        'face_found':         face_found,
        'face_quality':       face_quality,
        'explanation':        decision.get('explanation', ''),
        'signal_breakdown':   decision.get('signal_breakdown', {}),
        'is_uncertain':       decision.get('is_uncertain', True),
        'processing_time':    round(processing_time, 2),
        'error':              None,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_image_path(file_path):
    if not file_path:
        return "No file path provided."
    if not os.path.exists(str(file_path)):
        return f"File not found: {file_path}"
    ext = os.path.splitext(str(file_path))[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return f"Unsupported image format '{ext}'. Supported: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
    if os.path.getsize(str(file_path)) == 0:
        return "File is empty."
    return None


def _fallback_decision(cnn_result, forensic_result):
    """Minimal fallback if decision engine itself fails."""
    cnn_score      = cnn_result.get('cnn_score', 0.50)
    forensic_score = forensic_result.get('forensic_score', 0.40)
    final_score    = 0.55 * cnn_score + 0.45 * forensic_score
    return {
        'classification': 'UNCERTAIN',
        'label':          'Uncertain',
        'final_score':    round(final_score, 4),
        'confidence':     35.0,
        'explanation':    "Ensemble decision engine encountered an error. Analysis is inconclusive.",
        'signal_breakdown': {},
        'is_uncertain':   True,
    }


def _error_result(message, elapsed=0.0):
    return {
        'status':             'error',
        'cnn_score':          None,
        'cnn_mode':           'unavailable',
        'forensic_score':     None,
        'forensic_flags':     [],
        'provenance_status':  'unavailable',
        'provenance_markers': [],
        'provenance_message': '',
        'final_score':        0.5,
        'classification':     'UNCERTAIN',
        'label':              'Uncertain',
        'confidence':         0.0,
        'face_found':         False,
        'face_quality':       0.0,
        'explanation':        f"Analysis could not be completed: {message}",
        'signal_breakdown':   {},
        'is_uncertain':       True,
        'processing_time':    round(elapsed, 2),
        'error':              message,
    }
