"""
Video Detector Service
======================
Orchestrates the full video deepfake detection pipeline:

  1. Validate video
  2. Extract metadata
  3. Intelligent frame sampling
  4. Per-frame face detection + CNN inference
  5. Frame-level aggregation with temporal consistency check
  6. Provenance check on video file
  7. Ensemble decision
"""

import logging
import os
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Video settings
MAX_FRAMES_TO_ANALYZE = 60       # cap for efficiency
MIN_FRAMES_FOR_DECISION = 3      # need at least 3 good frames
FACE_FRAME_WEIGHT = 2.0          # face-containing frames count more
SUSPICIOUS_FRAME_THRESHOLD = 0.55  # per-frame CNN score above which = suspicious
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}


def detect_video(file_path, media_type='video', max_frames=MAX_FRAMES_TO_ANALYZE):
    """
    Full video deepfake detection pipeline.

    Args:
        file_path:   str — absolute path to video file
        media_type:  str — 'video', 'youtube', or 'instagram'
        max_frames:  int — maximum frames to sample

    Returns dict with same structure as image_detector.detect_image()
    plus video-specific fields.
    """
    t_start = time.monotonic()

    # ── Validate ──────────────────────────────────────────────────────────────
    validation_error = _validate_video_path(file_path)
    if validation_error:
        return _error_result(validation_error, time.monotonic() - t_start)

    logger.info("Video detection started: %s", os.path.basename(file_path))

    # ── Extract video metadata ────────────────────────────────────────────────
    meta = _extract_video_metadata(file_path)
    logger.info("Video metadata: %s", meta)

    # ── Frame sampling + per-frame analysis ──────────────────────────────────
    frame_results = _analyse_frames(file_path, max_frames)

    if not frame_results:
        return _uncertain_result(
            "No frames could be analysed from this video. "
            "The video may be empty, corrupt, or contain no readable frames.",
            meta,
            time.monotonic() - t_start,
        )

    # ── Aggregate frame-level predictions ────────────────────────────────────
    aggregated = _aggregate_frame_results(frame_results)
    logger.info(
        "Video aggregation: frames=%d face_frames=%d suspicious=%d "
        "weighted_fake_score=%.3f temporal_consistency=%.3f",
        aggregated['frames_analyzed'],
        aggregated['frames_with_face'],
        aggregated['suspicious_frames'],
        aggregated['weighted_fake_score'],
        aggregated['temporal_consistency'],
    )

    if aggregated['frames_analyzed'] < MIN_FRAMES_FOR_DECISION:
        return _uncertain_result(
            f"Only {aggregated['frames_analyzed']} frame(s) could be analysed. "
            "Insufficient data for a reliable prediction.",
            meta,
            time.monotonic() - t_start,
        )

    # Build CNN result dict from video aggregation (for decision engine)
    cnn_score = aggregated['weighted_fake_score']
    cnn_result = {
        'status':    'ok',
        'cnn_score': cnn_score,
        'mode':      aggregated.get('cnn_mode', 'heuristic'),
        'face_used': aggregated['frames_with_face'] > 0,
        'error':     None,
    }

    # ── Forensic analysis on a sample frame ──────────────────────────────────
    forensic_result = _forensic_on_sample_frame(file_path)

    # ── Provenance check ──────────────────────────────────────────────────────
    try:
        from .provenance.provenance_service import check_provenance
        provenance_result = check_provenance(file_path, media_type=media_type)
    except Exception as exc:
        logger.warning("Provenance check error: %s", exc)
        provenance_result = {'status': 'unavailable', 'markers': [], 'message': str(exc)}

    # ── Ensemble decision ─────────────────────────────────────────────────────
    video_result_for_engine = {
        'status':               'ok',
        'frames_analyzed':      aggregated['frames_analyzed'],
        'suspicious_frames':    aggregated['suspicious_frames'],
        'fake_frame_pct':       aggregated['fake_frame_pct'],
        'temporal_consistency': aggregated['temporal_consistency'],
    }
    try:
        from .ensemble.decision_engine import decide
        decision = decide(
            cnn_result=cnn_result,
            forensic_result=forensic_result,
            provenance_result=provenance_result,
            video_result=video_result_for_engine,
            face_found=aggregated['frames_with_face'] > 0,
            cnn_mode=aggregated.get('cnn_mode', 'heuristic'),
        )
    except Exception as exc:
        logger.error("Ensemble decision error: %s", exc)
        decision = {
            'classification': 'UNCERTAIN',
            'label': 'Uncertain',
            'final_score': cnn_score,
            'confidence': 35.0,
            'explanation': "Ensemble decision engine encountered an error.",
            'signal_breakdown': {},
            'is_uncertain': True,
        }

    processing_time = time.monotonic() - t_start
    logger.info(
        "Video detection complete: %s (%.1f%%) in %.2fs",
        decision.get('classification'), decision.get('confidence'), processing_time
    )

    return {
        'status':             'ok',
        'cnn_score':          cnn_score,
        'cnn_mode':           aggregated.get('cnn_mode', 'heuristic'),
        'forensic_score':     forensic_result.get('forensic_score'),
        'forensic_flags':     forensic_result.get('flags', []),
        'provenance_status':  provenance_result.get('status', 'unavailable'),
        'provenance_markers': provenance_result.get('markers', []),
        'provenance_message': provenance_result.get('message', ''),
        'final_score':        decision.get('final_score', cnn_score),
        'classification':     decision.get('classification', 'UNCERTAIN'),
        'label':              decision.get('label', 'Uncertain'),
        'confidence':         decision.get('confidence', 35.0),
        'face_found':         aggregated['frames_with_face'] > 0,
        'face_quality':       0.0,
        'explanation':        decision.get('explanation', ''),
        'signal_breakdown':   decision.get('signal_breakdown', {}),
        'is_uncertain':       decision.get('is_uncertain', True),
        # Video-specific
        'frames_analyzed':    aggregated['frames_analyzed'],
        'frames_with_face':   aggregated['frames_with_face'],
        'suspicious_frames':  aggregated['suspicious_frames'],
        'fake_frame_pct':     aggregated['fake_frame_pct'],
        'max_frame_score':    aggregated['max_frame_score'],
        'temporal_consistency': aggregated['temporal_consistency'],
        'video_metadata':     meta,
        'processing_time':    round(processing_time, 2),
        'error':              None,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Frame-level analysis
# ─────────────────────────────────────────────────────────────────────────────

def _analyse_frames(video_path, max_frames):
    """Sample frames and run per-frame CNN + face detection."""
    from .face_detector import detect_face
    from .cnn_detector import cnn_detector

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Could not open video: %s", video_path)
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 99999

    # Determine sampling interval — spread evenly
    interval = max(1, total_frames // max_frames)

    frame_results = []
    frame_idx = 0
    analyzed = 0
    cnn_mode = 'heuristic'

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0 and analyzed < max_frames:
            try:
                face_res = detect_face(frame)
                face_found = face_res.get('found', False)
                face_crop  = face_res.get('crop')

                # CNN on face crop if available, else full frame; classifier runs on full frame
                cnn_res = cnn_detector.predict_image(frame, face_crop if face_found else None)

                if cnn_res.get('status') == 'ok':
                    cnn_mode = cnn_res.get('mode', cnn_mode)
                    frame_results.append({
                        'frame_idx':  frame_idx,
                        'cnn_score':  cnn_res['cnn_score'],
                        'face_found': face_found,
                        'face_quality': face_res.get('quality', 0.0),
                    })
                    analyzed += 1

            except Exception as exc:
                logger.debug("Frame %d analysis error: %s", frame_idx, exc)

        frame_idx += 1

    cap.release()

    # Attach mode to each result (simplify aggregation)
    for r in frame_results:
        r['cnn_mode'] = cnn_mode

    logger.debug("Frame analysis complete: %d/%d frames processed", len(frame_results), analyzed)
    return frame_results


def _aggregate_frame_results(frame_results):
    """Aggregate per-frame predictions into video-level metrics."""
    if not frame_results:
        return _empty_aggregation()

    cnn_mode = frame_results[0].get('cnn_mode', 'heuristic')

    # Weighted sum: face frames get FACE_FRAME_WEIGHT × weight
    weighted_scores = []
    weights = []
    suspicious_frames = 0
    frames_with_face = 0
    all_scores = []

    for r in frame_results:
        score = r['cnn_score']
        face  = r['face_found']
        w     = FACE_FRAME_WEIGHT if face else 1.0

        weighted_scores.append(score * w)
        weights.append(w)
        all_scores.append(score)

        if face:
            frames_with_face += 1
        if score >= SUSPICIOUS_FRAME_THRESHOLD:
            suspicious_frames += 1

    total_w = sum(weights)
    weighted_fake_score = sum(weighted_scores) / total_w if total_w > 0 else 0.5

    # Median score (robust to outlier frames)
    median_score = float(np.median(all_scores))

    # Max suspicious score
    max_frame_score = float(max(all_scores))

    # Temporal consistency: low std → consistent predictions → higher confidence
    score_std = float(np.std(all_scores))
    # Map std to consistency: std=0 → 1.0, std=0.5 → 0.0
    temporal_consistency = float(np.clip(1.0 - score_std * 2.0, 0.0, 1.0))

    # Fake frame percentage among face-containing frames
    face_frame_scores = [r['cnn_score'] for r in frame_results if r['face_found']]
    if face_frame_scores:
        fake_face_frames = sum(1 for s in face_frame_scores if s >= SUSPICIOUS_FRAME_THRESHOLD)
        fake_frame_pct = (fake_face_frames / len(face_frame_scores)) * 100.0
    else:
        fake_frame_pct = (suspicious_frames / len(frame_results)) * 100.0 if frame_results else 0.0

    # Use a blend of weighted mean and median for robustness
    # This avoids a single outlier frame dominating the result
    blended_score = 0.65 * weighted_fake_score + 0.35 * median_score

    return {
        'frames_analyzed':     len(frame_results),
        'frames_with_face':    frames_with_face,
        'suspicious_frames':   suspicious_frames,
        'weighted_fake_score': float(np.clip(blended_score, 0.0, 1.0)),
        'max_frame_score':     max_frame_score,
        'fake_frame_pct':      round(fake_frame_pct, 1),
        'temporal_consistency': temporal_consistency,
        'cnn_mode':            cnn_mode,
    }


def _empty_aggregation():
    return {
        'frames_analyzed': 0, 'frames_with_face': 0, 'suspicious_frames': 0,
        'weighted_fake_score': 0.5, 'max_frame_score': 0.0,
        'fake_frame_pct': 0.0, 'temporal_consistency': 0.0, 'cnn_mode': 'heuristic',
    }


def _forensic_on_sample_frame(video_path):
    """Run forensic analysis on one sample frame from the video."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Sample at 10% through the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 10))
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return {'status': 'error', 'forensic_score': 0.40, 'flags': [], 'error': 'Could not extract sample frame'}

        from . import forensic_detector
        return forensic_detector.analyse(frame)
    except Exception as exc:
        logger.warning("Forensic on sample frame failed: %s", exc)
        return {'status': 'error', 'forensic_score': 0.40, 'flags': [], 'error': str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
#  Video metadata
# ─────────────────────────────────────────────────────────────────────────────

def _extract_video_metadata(video_path):
    """Extract basic video metadata via OpenCV."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {}
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps   = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frames / fps if fps > 0 else 0
        cap.release()

        try:
            file_size = os.path.getsize(str(video_path))
        except Exception:
            file_size = 0

        return {
            'width': w,
            'height': h,
            'fps': round(fps, 2),
            'frame_count': frames,
            'duration_seconds': round(duration, 2),
            'file_size': file_size,
        }
    except Exception as exc:
        logger.debug("Metadata extraction failed: %s", exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
#  Validation + error helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_video_path(file_path):
    if not file_path:
        return "No video file path provided."
    if not os.path.exists(str(file_path)):
        return f"Video file not found: {file_path}"
    if os.path.getsize(str(file_path)) == 0:
        return "Video file is empty."
    return None


def _uncertain_result(message, meta, elapsed):
    return {
        'status':             'ok',
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
        'explanation':        message,
        'signal_breakdown':   {},
        'is_uncertain':       True,
        'frames_analyzed':    0,
        'frames_with_face':   0,
        'suspicious_frames':  0,
        'fake_frame_pct':     0.0,
        'max_frame_score':    0.0,
        'temporal_consistency': 0.0,
        'video_metadata':     meta,
        'processing_time':    round(elapsed, 2),
        'error':              None,
    }


def _error_result(message, elapsed):
    result = _uncertain_result(message, {}, elapsed)
    result['status'] = 'error'
    result['error']  = message
    result['confidence'] = 0.0
    return result
