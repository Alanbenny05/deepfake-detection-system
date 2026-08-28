"""
Forensic Detector Module
========================
Image forensic analysis for deepfake detection.

Analyses image characteristics that differ between real camera photographs
and AI-generated / manipulated images, using signal-processing techniques.

All signals are grounded in published forensic literature:
  1. Error Level Analysis (ELA) — JPEG compression inconsistency
  2. DCT frequency spectrum — GAN checkerboard/periodic artefacts
  3. Local noise uniformity — real sensors have spatially variant noise
  4. Colour saturation profile — GANs produce abnormal saturation patterns
  5. Laplacian blur/over-sharpening signature

Returns a structured dict — never modifies the input image.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Individual forensic signals
# ─────────────────────────────────────────────────────────────────────────────

def _ela_score(bgr_img, quality=90):
    """
    Error Level Analysis score.

    Re-saves the image at a fixed JPEG quality and computes pixel-level
    differences. Real images have spatially consistent compression error;
    manipulated regions show inconsistent error patterns.

    Returns float in [0, 1] where higher = more suspicious.
    """
    try:
        import io
        from PIL import Image as PILImage

        # Convert BGR → RGB → PIL
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        # Re-save at reference quality
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        recompressed = np.array(PILImage.open(buf).convert('RGB'))
        recompressed_bgr = cv2.cvtColor(recompressed, cv2.COLOR_RGB2BGR)

        # Compute normalised absolute error
        diff = np.abs(bgr_img.astype(np.float32) - recompressed_bgr.astype(np.float32))
        mean_error = float(np.mean(diff))
        std_error  = float(np.std(diff))

        # High mean OR very high std → suspicious
        # Typical real JPEG: mean≈5–20, std≈3–15
        if mean_error > 30 or std_error > 25:
            return 0.70
        elif mean_error > 15 or std_error > 12:
            return 0.45
        else:
            return 0.25
    except Exception as exc:
        logger.debug("ELA analysis skipped: %s", exc)
        return 0.40  # neutral if unavailable


def _dct_frequency_score(gray_img):
    """
    DCT high-frequency energy ratio.

    GANs often leave periodic high-frequency artefacts (checkerboard patterns
    from transposed convolutions). Elevated high-frequency energy is suspicious.

    Returns float in [0, 1].
    """
    try:
        gray_small = cv2.resize(gray_img, (64, 64)).astype(np.float32)
        dct = cv2.dct(gray_small)
        total_energy   = float(np.sum(dct ** 2)) + 1e-9
        low_freq_energy = float(np.sum(dct[:8, :8] ** 2))
        high_freq_ratio = 1.0 - (low_freq_energy / total_energy)

        if high_freq_ratio > 0.60:
            return 0.70
        elif high_freq_ratio > 0.45:
            return 0.45
        else:
            return 0.25
    except Exception as exc:
        logger.debug("DCT analysis failed: %s", exc)
        return 0.40


def _noise_uniformity_score(gray_img):
    """
    Local noise variance uniformity.

    Real camera images exhibit spatially varying shot noise (higher ISO →
    higher variance in low-light regions). GAN images tend to be more
    globally uniform in local noise.

    Returns float in [0, 1].
    """
    try:
        h, w = gray_img.shape[:2]
        if h < 16 or w < 16:
            return 0.40  # too small to analyse

        ph, pw = max(1, h // 4), max(1, w // 4)
        local_stds = []
        for py in range(4):
            for px in range(4):
                patch = gray_img[py*ph:(py+1)*ph, px*pw:(px+1)*pw]
                local_stds.append(float(np.std(patch.astype(np.float32))))

        std_of_stds = float(np.std(local_stds))

        if std_of_stds < 3.0:
            return 0.65      # very uniform noise → GAN-like
        elif std_of_stds < 8.0:
            return 0.40
        else:
            return 0.25
    except Exception as exc:
        logger.debug("Noise analysis failed: %s", exc)
        return 0.40


def _saturation_score(bgr_img):
    """
    Colour saturation profile.

    GANs sometimes produce abnormally high or low saturation, or unusually
    uniform saturation across the image.

    Returns float in [0, 1].
    """
    try:
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        sat_mean = float(np.mean(sat))
        sat_std  = float(np.std(sat))

        if sat_mean > 180 or sat_std < 15:
            return 0.65
        elif sat_mean < 20:
            return 0.60
        else:
            return 0.28
    except Exception as exc:
        logger.debug("Saturation analysis failed: %s", exc)
        return 0.40


def _sharpness_score(gray_img):
    """
    Laplacian variance — sharpness / smoothing signature.

    GAN images are often over-smoothed (very low Laplacian) or artificially
    over-sharpened (very high). Natural range ≈ 80–600.

    Returns float in [0, 1].
    """
    try:
        lap_var = float(cv2.Laplacian(gray_img, cv2.CV_64F).var())

        if lap_var < 30:
            return 0.78      # extreme smoothing → GAN-like
        elif lap_var < 80:
            return 0.60
        elif lap_var < 500:
            return 0.28      # natural sharpness
        else:
            return 0.20      # very sharp → likely real
    except Exception as exc:
        logger.debug("Sharpness analysis failed: %s", exc)
        return 0.40


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

#  Signal weights (must sum to 1.0)
#  Documented here for easy adjustment:
#    ELA       — most discriminative for splicing / recompression artefacts
#    Sharpness — reliable indicator of GAN over-smoothing
#    DCT       — catches GAN periodic patterns
#    Noise     — secondary signal
#    Saturation — secondary signal
_WEIGHTS = {
    'ela':        0.28,
    'sharpness':  0.25,
    'dct':        0.22,
    'noise':      0.15,
    'saturation': 0.10,
}


def analyse(bgr_img):
    """
    Run all forensic signals on a BGR image.

    Args:
        bgr_img: numpy BGR array

    Returns dict:
        {
            'status':            'ok' | 'error',
            'forensic_score':    float [0,1] (weighted combination),
            'ela_score':         float,
            'dct_score':         float,
            'noise_score':       float,
            'saturation_score':  float,
            'sharpness_score':   float,
            'flags':             list[str],   # human-readable suspicious observations
            'error':             str or None,
        }
    """
    if bgr_img is None or bgr_img.size == 0:
        return _error_result("Empty image supplied to forensic analyser")

    try:
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        ela_s  = _ela_score(bgr_img)
        dct_s  = _dct_frequency_score(gray)
        noise_s = _noise_uniformity_score(gray)
        sat_s  = _saturation_score(bgr_img)
        sharp_s = _sharpness_score(gray)

        # Weighted combination
        combined = (
            _WEIGHTS['ela']        * ela_s  +
            _WEIGHTS['dct']        * dct_s  +
            _WEIGHTS['noise']      * noise_s +
            _WEIGHTS['saturation'] * sat_s  +
            _WEIGHTS['sharpness']  * sharp_s
        )
        combined = float(np.clip(combined, 0.0, 1.0))

        # Build human-readable flags for UI
        flags = []
        if ela_s > 0.55:
            flags.append("JPEG compression inconsistency detected")
        if dct_s > 0.55:
            flags.append("Unusual high-frequency spectral patterns")
        if noise_s > 0.55:
            flags.append("Abnormally uniform local noise profile")
        if sat_s > 0.55:
            flags.append("Unusual colour saturation characteristics")
        if sharp_s > 0.55:
            flags.append("Over-smoothing or artificial sharpening detected")

        logger.debug(
            "Forensic: ela=%.2f dct=%.2f noise=%.2f sat=%.2f sharp=%.2f → combined=%.3f",
            ela_s, dct_s, noise_s, sat_s, sharp_s, combined
        )

        return {
            'status':           'ok',
            'forensic_score':   combined,
            'ela_score':        ela_s,
            'dct_score':        dct_s,
            'noise_score':      noise_s,
            'saturation_score': sat_s,
            'sharpness_score':  sharp_s,
            'flags':            flags,
            'error':            None,
        }

    except Exception as exc:
        logger.warning("Forensic analysis failed: %s", exc)
        return _error_result(str(exc))


def _error_result(reason):
    return {
        'status':           'error',
        'forensic_score':   0.40,   # neutral — not penalising for analysis failure
        'ela_score':        0.40,
        'dct_score':        0.40,
        'noise_score':      0.40,
        'saturation_score': 0.40,
        'sharpness_score':  0.40,
        'flags':            [],
        'error':            reason,
    }
