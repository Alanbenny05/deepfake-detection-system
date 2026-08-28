"""
SynthID / AI-Provenance Detector
=================================
Checks images and videos for metadata-level provenance signals indicating
AI generation or manipulation.

IMPORTANT — What this module does and does NOT do:
  - It reads EXIF, IPTC, XMP, and file metadata for known AI-generation markers.
  - It does NOT have access to Google SynthID's proprietary watermark detection
    API (no public general-purpose API exists for arbitrary uploads as of 2026).
  - Absence of a detected provenance marker does NOT mean the media is real.
  - Results are supplementary signals only — not proof of authenticity or fakery.

Provenance markers checked:
  1. EXIF Software / Creator fields mentioning known AI tools
  2. XMP dc:creator / photoshop:Source / exif:UserComment with AI mentions
  3. C2PA / JUMBF content credential blocks (detected by presence in JPEG SOI)
  4. Stable Diffusion / Midjourney / DALL-E metadata fingerprints
  5. PNG tEXt chunks with AI generation parameters
"""

import logging
import os
import re
import struct

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Known AI generation software signatures
# ─────────────────────────────────────────────────────────────────────────────

_AI_SOFTWARE_PATTERNS = [
    r'stable.?diffusion',
    r'midjourney',
    r'dall[\-\s]?e',
    r'runway',
    r'adobe.?firefly',
    r'ideogram',
    r'leonardo',
    r'imageai',
    r'bing.?image',
    r'canva.?ai',
    r'dream.?studio',
    r'automatic1111',
    r'comfyui',
    r'invokeai',
    r'novelai',
    r'seaart',
    r'wombodream',
    r'nightcafe',
    r'artbreeder',
    r'civitai',
    r'diffusion',
    r'gan.*generated',
    r'ai.?generated',
    r'synthid',         # if Google embeds this in metadata
]

_AI_PATTERN_RE = re.compile(
    '|'.join(_AI_SOFTWARE_PATTERNS),
    re.IGNORECASE
)

# C2PA / JUMBF marker in JPEG (content credentials)
_C2PA_MARKER = b'jumbf'
_C2PA_XMP_MARKER = b'http://ns.adobe.com/xap'


# ─────────────────────────────────────────────────────────────────────────────
#  EXIF reader (no piexif required — uses PIL which is already a dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _read_exif_strings(file_path):
    """Return a list of text values from EXIF/IPTC/XMP using PIL."""
    strings = []
    try:
        from PIL import Image as PILImage
        from PIL.ExifTags import TAGS

        with PILImage.open(file_path) as img:
            # Basic EXIF
            exif_data = img._getexif() if hasattr(img, '_getexif') else None
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, str(tag_id))
                    if isinstance(value, (str, bytes)):
                        val_str = value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value
                        strings.append(f"{tag}:{val_str}")

            # PNG text chunks
            if img.format == 'PNG':
                info = img.info or {}
                for k, v in info.items():
                    strings.append(f"{k}:{v}")

            # XMP
            xmp_data = img.info.get('XML:com.adobe.xmp') or img.info.get('xmp') or ''
            if xmp_data:
                strings.append(str(xmp_data))

    except Exception as exc:
        logger.debug("EXIF read error for %s: %s", file_path, exc)

    return strings


def _read_raw_bytes(file_path, max_bytes=65536):
    """Read initial bytes of file for binary marker search."""
    try:
        with open(file_path, 'rb') as f:
            return f.read(max_bytes)
    except Exception:
        return b''


# ─────────────────────────────────────────────────────────────────────────────
#  Detection logic
# ─────────────────────────────────────────────────────────────────────────────

def check_image_provenance(file_path):
    """
    Check an image file for AI-generation metadata markers.

    Returns dict:
        {
            'status':   'detected' | 'not_detected' | 'unavailable' | 'error',
            'provider': str or None,
            'markers':  list[str],   # human-readable found markers
            'message':  str,
        }
    """
    if not file_path or not os.path.exists(str(file_path)):
        return _unavailable("File not found or path invalid")

    try:
        found_markers = []

        # 1. EXIF / PNG text string scan
        exif_strings = _read_exif_strings(file_path)
        for s in exif_strings:
            if _AI_PATTERN_RE.search(s):
                found_markers.append(f"Metadata field: {s[:120]}")

        # 2. Raw binary C2PA / JUMBF content credentials
        raw = _read_raw_bytes(file_path)
        if _C2PA_MARKER in raw.lower():
            found_markers.append("C2PA/JUMBF content credentials block detected")
        if _C2PA_XMP_MARKER in raw:
            # Scan the XMP payload for AI mentions
            xmp_idx = raw.find(_C2PA_XMP_MARKER)
            xmp_snippet = raw[xmp_idx:xmp_idx+4096].decode('utf-8', errors='replace')
            if _AI_PATTERN_RE.search(xmp_snippet):
                found_markers.append("XMP metadata contains AI generation reference")

        # 3. Stable Diffusion PNG parameter chunk
        if raw[:8] == b'\x89PNG\r\n\x1a\n':
            # PNG — check tEXt / iTXt chunks
            if b'parameters' in raw and (
                b'Negative prompt' in raw or b'Steps:' in raw or b'Sampler:' in raw
            ):
                found_markers.append("Stable Diffusion PNG generation parameters found")

        if found_markers:
            logger.info("Provenance markers detected in %s: %s", file_path, found_markers)
            return {
                'status':   'detected',
                'provider': 'Metadata/C2PA',
                'markers':  found_markers,
                'message':  (
                    f"AI-generation metadata found ({len(found_markers)} marker(s)). "
                    "This indicates the image was likely created or processed by an AI tool."
                ),
            }
        else:
            logger.debug("No provenance markers found in %s", file_path)
            return {
                'status':   'not_detected',
                'provider': None,
                'markers':  [],
                'message':  (
                    "No AI-generation metadata markers were found in this image. "
                    "Note: absence of metadata markers does not confirm authenticity — "
                    "metadata can be stripped or may not be present in all AI-generated images."
                ),
            }

    except Exception as exc:
        logger.warning("Provenance check error for %s: %s", file_path, exc)
        return _error(str(exc))


def check_video_provenance(file_path):
    """
    Check a video file for AI-generation provenance.

    Note: Video metadata provenance is more limited than images. We check
    encoding tool strings and container metadata where possible.

    Returns same structure as check_image_provenance().
    """
    if not file_path or not os.path.exists(str(file_path)):
        return _unavailable("Video file not found")

    try:
        found_markers = []

        # Check raw container header for AI tool markers
        raw = _read_raw_bytes(file_path, max_bytes=32768)
        raw_str = raw.decode('utf-8', errors='replace')

        if _AI_PATTERN_RE.search(raw_str):
            found_markers.append("AI tool reference in video container metadata")

        # Try reading with OpenCV for basic codec info (non-destructive)
        try:
            import cv2
            cap = cv2.VideoCapture(str(file_path))
            # No metadata fields available directly via OpenCV, but we confirm openability
            cap.release()
        except Exception:
            pass

        if found_markers:
            return {
                'status':   'detected',
                'provider': 'Container metadata',
                'markers':  found_markers,
                'message':  f"AI-generation references found in video metadata ({len(found_markers)} marker(s)).",
            }

        # For video, detailed provenance is unavailable without specialised tooling
        return {
            'status':   'unavailable',
            'provider': None,
            'markers':  [],
            'message':  (
                "Detailed provenance verification for video content is not available "
                "without access to specialised content authentication tools. "
                "The analysis relies on CNN and forensic signals."
            ),
        }

    except Exception as exc:
        logger.warning("Video provenance check error for %s: %s", file_path, exc)
        return _error(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unavailable(reason):
    return {
        'status':   'unavailable',
        'provider': None,
        'markers':  [],
        'message':  f"Provenance check unavailable: {reason}",
    }


def _error(reason):
    return {
        'status':   'error',
        'provider': None,
        'markers':  [],
        'message':  f"Provenance check encountered an error: {reason}",
    }
