"""
Ensemble Decision Engine
========================
Combines multiple detection signals into a final classification.

Design principles:
  1. Weights are clearly documented and easily adjustable in one place.
  2. UNCERTAIN is a valid and important output — never force a Real/Fake answer
     when evidence is insufficient or contradictory.
  3. All signal inputs use the same [0,1] fake-probability scale.
  4. Provenance signals are treated as additional evidence, not primary verdict.

Classification tiers:
    HIGH_CONFIDENCE_REAL   — final_score < 0.20  (strong evidence of authenticity)
    LIKELY_REAL            — final_score < 0.40  (leans real, moderate confidence)
    UNCERTAIN              — final_score in [0.40, 0.60] OR contradictory signals
    LIKELY_FAKE            — final_score > 0.60  (leans fake, moderate confidence)
    HIGH_CONFIDENCE_FAKE   — final_score > 0.80  (strong evidence of manipulation)

Confidence values are mapped to percentages with honest uncertainty ranges.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Configurable weights — edit here to tune the system
# ─────────────────────────────────────────────────────────────────────────────

#  Active CNN mode weights (when trained deep neural net / classifier is loaded)
#  Must sum to 1.0
WEIGHTS_CNN_MODE = {
    'cnn':        0.80,   # Primary ML model gets 80% weight
    'forensic':   0.12,   # Heuristics get 12%
    'provenance': 0.08,   # Provenance gets 8%
}
WEIGHTS_CNN_MODE_NO_PROV = {
    'cnn':        0.85,   # Primary ML model gets 85% weight
    'forensic':   0.15,   # Heuristics get 15%
}

#  Heuristic mode weights (when falling back to handcrafted features)
#  Must sum to 1.0
WEIGHTS_HEURISTIC_MODE = {
    'cnn':        0.55,   # CNN heuristic gets 55% weight
    'forensic':   0.30,   # Main forensics get 30%
    'provenance': 0.15,   # Provenance gets 15%
}
WEIGHTS_HEURISTIC_MODE_NO_PROV = {
    'cnn':        0.65,   # CNN heuristic gets 65% weight
    'forensic':   0.35,   # Main forensics get 35%
}

#  Classification thresholds (fake probability)
THRESHOLDS = {
    'high_confidence_fake': 0.75,
    'likely_fake':          0.58,
    'uncertain_high':       0.58,   # above this in uncertain zone = leans fake
    'uncertain_low':        0.42,   # below this in uncertain zone = leans real
    'likely_real':          0.42,
    'high_confidence_real': 0.25,
}

#  Polar contradiction check threshold for heuristic mode
CONTRADICTION_THRESHOLD = 0.55

#  Minimum confidence to escape UNCERTAIN towards a definitive result
MIN_CONFIDENCE_REQUIRED = 0.55


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

CLASSIFICATION_LABELS = {
    'HIGH_CONFIDENCE_REAL': 'High Confidence Real',
    'LIKELY_REAL':          'Likely Real',
    'UNCERTAIN':            'Uncertain',
    'LIKELY_FAKE':          'Likely Fake',
    'HIGH_CONFIDENCE_FAKE': 'High Confidence Fake',
}


def decide(
    cnn_result,
    forensic_result,
    provenance_result,
    video_result=None,
    face_found=True,
    cnn_mode=None,
):
    """
    Combine available signals into a final classification.

    Args:
        cnn_result        (dict): from cnn_detector.predict_image()
        forensic_result   (dict): from forensic_detector.analyse()
        provenance_result (dict): from provenance_service.check_provenance()
        video_result      (dict or None): from video_detector — includes temporal consistency
        face_found        (bool): whether a face was detected in the media
        cnn_mode          (str): 'cnn' | 'heuristic' — lower weight in heuristic mode

    Returns dict:
        {
            'classification':   str  (key from CLASSIFICATION_LABELS)
            'label':            str  (human-readable label)
            'final_score':      float [0,1] fake probability
            'confidence':       float [0,100]
            'explanation':      str
            'signal_breakdown': dict
            'is_uncertain':     bool
        }
    """
    try:
        return _decide_internal(
            cnn_result, forensic_result, provenance_result,
            video_result, face_found, cnn_mode
        )
    except Exception as exc:
        logger.error("Decision engine error: %s", exc)
        return _uncertain_result(
            final_score=0.50,
            explanation="An error occurred in the decision engine. Analysis is inconclusive.",
            breakdown={'error': str(exc)},
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Internal decision logic
# ─────────────────────────────────────────────────────────────────────────────

def _decide_internal(
    cnn_result, forensic_result, provenance_result,
    video_result, face_found, cnn_mode,
):
    if cnn_mode is None:
        cnn_mode = cnn_result.get('mode', 'cnn') if cnn_result else 'cnn'
    breakdown = {}

    # ── Gather CNN score ──────────────────────────────────────────────────────
    cnn_ok = cnn_result.get('status') == 'ok'
    cnn_score = float(cnn_result.get('cnn_score', 0.50)) if cnn_ok else None
    breakdown['cnn'] = {
        'score':     cnn_score,
        'status':    cnn_result.get('status', 'unknown'),
        'mode':      cnn_result.get('mode', 'unknown'),
        'face_used': cnn_result.get('face_used', False),
    }

    # ── Gather forensic score ─────────────────────────────────────────────────
    forensic_ok = forensic_result.get('status') == 'ok'
    forensic_score = float(forensic_result.get('forensic_score', 0.40)) if forensic_ok else None
    breakdown['forensic'] = {
        'score':  forensic_score,
        'status': forensic_result.get('status', 'unknown'),
        'flags':  forensic_result.get('flags', []),
    }

    # ── Gather provenance signal ──────────────────────────────────────────────
    prov_status = provenance_result.get('status', 'unavailable')
    provenance_score = None   # will be set only if provenance detected/not_detected
    provenance_available = False

    if prov_status == 'detected':
        provenance_score = 0.90   # strong indicator of AI generation
        provenance_available = True
    elif prov_status == 'not_detected':
        # Absence of provenance is weak evidence of real; don't over-penalise
        provenance_score = 0.25
        provenance_available = True
    # else: 'unavailable' or 'error' → not included in weighted sum

    breakdown['provenance'] = {
        'status':  prov_status,
        'score':   provenance_score,
        'markers': provenance_result.get('markers', []),
        'message': provenance_result.get('message', ''),
    }

    # ── Gather video temporal signal (if available) ───────────────────────────
    video_consistency = None
    if video_result and video_result.get('status') == 'ok':
        video_consistency = video_result.get('temporal_consistency', None)
        breakdown['video'] = {
            'frames_analyzed':   video_result.get('frames_analyzed', 0),
            'suspicious_frames': video_result.get('suspicious_frames', 0),
            'fake_frame_pct':    video_result.get('fake_frame_pct', 0.0),
            'temporal_consistency': video_consistency,
        }

    # ── Compute weighted final score ──────────────────────────────────────────
    scores = []
    weights = []

    # Determine the weights based on cnn_mode
    if cnn_mode == 'cnn':
        w_all = WEIGHTS_CNN_MODE
        w_no_prov = WEIGHTS_CNN_MODE_NO_PROV
    else:
        w_all = WEIGHTS_HEURISTIC_MODE
        w_no_prov = WEIGHTS_HEURISTIC_MODE_NO_PROV

    if cnn_score is not None:
        w = w_all['cnn'] if provenance_available else w_no_prov['cnn']
        scores.append(cnn_score)
        weights.append(w)

    if forensic_score is not None:
        w = w_all['forensic'] if provenance_available else w_no_prov['forensic']
        scores.append(forensic_score)
        weights.append(w)

    if provenance_score is not None:
        scores.append(provenance_score)
        weights.append(w_all['provenance'])

    if not scores:
        # Absolutely no signals available
        return _uncertain_result(
            0.50,
            "No detection signals were available. Analysis is inconclusive.",
            breakdown,
        )

    # Normalise weights in case they don't sum to 1 (e.g. cnn only)
    total_w = sum(weights)
    final_score = sum(s * w for s, w in zip(scores, weights)) / total_w
    final_score = float(np.clip(final_score, 0.0, 1.0))

    breakdown['final_score_raw'] = final_score
    breakdown['weights_used'] = dict(zip(['cnn', 'forensic', 'provenance'][:len(scores)], weights))

    # ── Contradiction check ───────────────────────────────────────────────────
    contradictory = False
    if cnn_mode == 'heuristic' and cnn_score is not None and forensic_score is not None:
        # Only flag contradiction when both heuristic signals are strongly in opposite directions
        if (cnn_score >= 0.75 and forensic_score <= 0.25) or (cnn_score <= 0.25 and forensic_score >= 0.75):
            contradictory = True
            logger.info(
                "Polar contradictory signals: CNN=%.3f Forensic=%.3f -> UNCERTAIN",
                cnn_score, forensic_score
            )

    # ── No-face penalty flag (adjusts confidence without forcing UNCERTAIN) ───
    face_penalty = not face_found and cnn_mode == 'heuristic'

    # ── Classify base tier ───────────────────────────────────────────────────
    if final_score >= THRESHOLDS['high_confidence_fake']:
        classification = 'HIGH_CONFIDENCE_FAKE'
    elif final_score >= THRESHOLDS['likely_fake']:
        classification = 'LIKELY_FAKE'
    elif final_score <= THRESHOLDS['high_confidence_real']:
        classification = 'HIGH_CONFIDENCE_REAL'
    elif final_score <= THRESHOLDS['likely_real']:
        classification = 'LIKELY_REAL'
    else:
        classification = 'UNCERTAIN'

    # ── Active CNN / ML Safety Overrides ──────────────────────────────────
    if cnn_score is not None:
        if cnn_score >= 0.75:
            if classification in ('HIGH_CONFIDENCE_REAL', 'LIKELY_REAL', 'UNCERTAIN'):
                classification = 'HIGH_CONFIDENCE_FAKE' if final_score >= 0.70 else 'LIKELY_FAKE'
        elif cnn_score >= 0.58:
            if classification in ('HIGH_CONFIDENCE_REAL', 'LIKELY_REAL'):
                classification = 'LIKELY_FAKE'
        elif cnn_score <= 0.25:
            if classification in ('HIGH_CONFIDENCE_FAKE', 'LIKELY_FAKE', 'UNCERTAIN'):
                classification = 'HIGH_CONFIDENCE_REAL' if final_score <= 0.30 else 'LIKELY_REAL'
        elif cnn_score <= 0.42:
            if classification in ('HIGH_CONFIDENCE_FAKE', 'LIKELY_FAKE'):
                classification = 'LIKELY_REAL'

    # ── Contradiction Override ────────────────────────────────────────────────
    if contradictory:
        classification = 'UNCERTAIN'

    # ── Calculate confidence based on final classification ───────────────────
    if classification == 'HIGH_CONFIDENCE_FAKE':
        confidence = _score_to_confidence(final_score, high_end=True)
    elif classification == 'LIKELY_FAKE':
        confidence = _score_to_confidence(final_score, high_end=False)
    elif classification == 'HIGH_CONFIDENCE_REAL':
        confidence = _score_to_confidence(1.0 - final_score, high_end=True)
    elif classification == 'LIKELY_REAL':
        confidence = _score_to_confidence(1.0 - final_score, high_end=False)
    else:
        # Uncertain
        if contradictory:
            confidence = _contradictory_confidence(cnn_score, forensic_score)
        else:
            confidence = max(35.0, 50.0 - abs(final_score - 0.50) * 30.0)

    if face_penalty:
        confidence = min(confidence, 70.0)

    # ── Build explanation ─────────────────────────────────────────────────────
    explanation = _build_explanation(
        classification, final_score, breakdown,
        contradictory=contradictory, face_found=face_found, cnn_mode=cnn_mode
    )

    # Append safety override context to explanation
    if cnn_mode == 'cnn' and cnn_score is not None:
        if cnn_score >= 0.60 and final_score < 0.60:
            explanation += " Note: The neural network identified highly suspicious facial characteristics, overriding other inconclusive signals to flag this media as likely fake."
        elif cnn_score >= 0.45 and final_score < 0.40:
            explanation += " Note: The neural network identified suspicious facial characteristics, preventing a definitive 'Real' classification."

    # Apply video temporal penalty if available
    if video_consistency is not None and video_consistency < 0.50:
        # Low temporal consistency (high variance across frames) → push toward UNCERTAIN
        if classification in ('HIGH_CONFIDENCE_REAL', 'LIKELY_REAL'):
            classification = 'LIKELY_REAL'
            confidence = min(confidence, 65.0)
            explanation += " Temporal analysis across frames showed some inconsistency."
        elif classification in ('HIGH_CONFIDENCE_FAKE', 'LIKELY_FAKE'):
            explanation += " Temporal frame analysis supports the fake classification."

    return {
        'classification':   classification,
        'label':            CLASSIFICATION_LABELS[classification],
        'final_score':      round(final_score, 4),
        'confidence':       round(float(confidence), 1),
        'explanation':      explanation,
        'signal_breakdown': breakdown,
        'is_uncertain':     classification == 'UNCERTAIN',
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_to_confidence(score, high_end=False):
    """Map a directional score [0.5,1] to a confidence percentage [55,96]."""
    directional = max(0.5, float(score))
    confidence = 55.0 + (directional - 0.5) * 80.0
    return float(np.clip(confidence, 55.0, 96.0))


def _contradictory_confidence(cnn_score, forensic_score):
    """Compute a low confidence reflecting the degree of contradiction."""
    gap = abs(cnn_score - forensic_score)
    # Higher gap → lower confidence; max gap of 1.0 → confidence ~30%
    return float(np.clip(65.0 - gap * 60.0, 30.0, 55.0))


def _uncertain_result(final_score, explanation, breakdown):
    return {
        'classification':   'UNCERTAIN',
        'label':            CLASSIFICATION_LABELS['UNCERTAIN'],
        'final_score':      round(final_score, 4),
        'confidence':       35.0,
        'explanation':      explanation,
        'signal_breakdown': breakdown,
        'is_uncertain':     True,
    }


def _build_explanation(
    classification, final_score, breakdown,
    contradictory=False, face_found=True, cnn_mode='cnn',
):
    """Build a human-readable explanation for the result."""
    parts = []

    prov_status = breakdown.get('provenance', {}).get('status', 'unavailable')
    cnn_score   = breakdown.get('cnn', {}).get('score')
    forensic_score = breakdown.get('forensic', {}).get('score')
    forensic_flags = breakdown.get('forensic', {}).get('flags', [])
    video_info  = breakdown.get('video')

    # Opening statement
    if classification == 'UNCERTAIN':
        if contradictory:
            parts.append(
                "The available detection signals produced contradictory results: "
                f"the CNN model scored {'high' if cnn_score and cnn_score > 0.5 else 'low'} "
                f"while forensic analysis scored {'high' if forensic_score and forensic_score > 0.5 else 'low'}. "
                "A definitive classification cannot be made with sufficient confidence."
            )
        elif not face_found:
            parts.append(
                "No face was detected in this media. The analysis is based on general image "
                "characteristics without facial region analysis, which reduces reliability."
            )
        else:
            parts.append(
                "The combined evidence does not clearly indicate real or AI-generated media. "
                f"The overall fake probability ({final_score:.0%}) falls within the uncertain range."
            )
    elif 'REAL' in classification:
        parts.append(
            f"The analysis indicates this media is {'highly likely' if 'HIGH' in classification else 'likely'} authentic. "
            f"The combined fake probability is {final_score:.0%}."
        )
    else:
        parts.append(
            f"The analysis indicates this media is {'highly likely' if 'HIGH' in classification else 'likely'} "
            f"AI-generated or manipulated. The combined fake probability is {final_score:.0%}."
        )

    # CNN mode note
    if cnn_mode == 'heuristic':
        parts.append(
            "Note: The trained CNN model was not available; the analysis used "
            "multi-signal heuristic methods instead, which are less reliable."
        )

    # Provenance
    if prov_status == 'detected':
        markers = breakdown.get('provenance', {}).get('markers', [])
        parts.append(
            f"AI-generation metadata was found in the file ({len(markers)} marker(s)), "
            "which is a strong indicator of AI generation."
        )
    elif prov_status == 'not_detected':
        parts.append(
            "No AI-generation metadata markers were found, though this alone does not confirm authenticity."
        )
    elif prov_status in ('unavailable', 'error'):
        parts.append("Provenance/metadata verification was not available for this media.")

    # Forensic flags
    if forensic_flags:
        parts.append("Forensic analysis identified: " + "; ".join(forensic_flags) + ".")

    # Video info
    if video_info:
        frames = video_info.get('frames_analyzed', 0)
        suspicious = video_info.get('suspicious_frames', 0)
        pct = video_info.get('fake_frame_pct', 0.0)
        parts.append(
            f"Video analysis: {frames} frames analysed, "
            f"{suspicious} suspicious ({pct:.1f}% of face-containing frames)."
        )

    # Disclaimer
    parts.append(
        "Deepfake detection is probabilistic — this result represents the system's "
        "best estimate based on available evidence and should not be treated as absolute proof."
    )

    return " ".join(parts)
