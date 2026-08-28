"""
Provenance Service
==================
Public entry point for provenance/content authentication checks.

Usage:
    from detector.provenance.provenance_service import check_provenance

    result = check_provenance(file_path, media_type='image')
    # result['status'] in {'detected', 'not_detected', 'unavailable', 'error'}
"""

import logging
import os

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}
_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}


def check_provenance(file_path, media_type=None):
    """
    Check provenance/content authentication for an image or video file.

    Args:
        file_path:  str — absolute path to the media file
        media_type: str — 'image', 'video', 'youtube', 'instagram', or None
                    (auto-detected from extension if None)

    Returns dict:
        {
            'status':   'detected' | 'not_detected' | 'unavailable' | 'error',
            'provider': str or None,
            'markers':  list[str],
            'message':  str,
        }

    IMPORTANT:
        - 'not_detected' means no markers were found.
          It does NOT mean the media is authentic.
        - 'unavailable' means the check could not be performed
          (e.g. file not accessible, video type).
        - 'detected' means AI-generation metadata was found.
    """
    if not file_path:
        return _unavailable("No file path provided")

    file_path = str(file_path)

    # Auto-detect media type from extension
    if media_type is None or media_type not in ('image', 'video', 'youtube', 'instagram'):
        ext = os.path.splitext(file_path)[1].lower()
        if ext in _IMAGE_EXTENSIONS:
            media_type = 'image'
        elif ext in _VIDEO_EXTENSIONS:
            media_type = 'video'
        else:
            media_type = 'unknown'

    if not os.path.exists(file_path):
        return _unavailable(f"File does not exist: {file_path}")

    try:
        from .synthid_detector import check_image_provenance, check_video_provenance

        if media_type == 'image':
            result = check_image_provenance(file_path)
        elif media_type in ('video', 'youtube', 'instagram'):
            result = check_video_provenance(file_path)
        else:
            result = _unavailable(f"Unknown media type: {media_type}")

        logger.info(
            "Provenance check complete for %s (type=%s): status=%s",
            os.path.basename(file_path), media_type, result.get('status')
        )
        return result

    except Exception as exc:
        logger.error("Provenance service error: %s", exc)
        return {
            'status':   'error',
            'provider': None,
            'markers':  [],
            'message':  f"Provenance service encountered an unexpected error: {exc}",
        }


def _unavailable(reason):
    return {
        'status':   'unavailable',
        'provider': None,
        'markers':  [],
        'message':  reason,
    }
