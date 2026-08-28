"""
Detection Service
=================
Thin orchestration layer called by Django views.

Responsibilities:
  - Route to image or video detector based on media type
  - Handle URL-sourced media (YouTube, Instagram)
  - Validate URLs with SSRF protection
  - Clean up temporary files after processing
  - Return a unified result dict regardless of input type
  - NEVER use random values as fallback

Usage (from views.py):
    from .detection_service import DetectionService

    service = DetectionService()
    result = service.analyse_file(file_path, media_type='image')
    result = service.analyse_url(url, platform='youtube', user=request.user)
"""

import logging
import os
import re
import tempfile
import time
import urllib.parse

logger = logging.getLogger(__name__)

# Allowed URL schemes (no file://, no internal network)
ALLOWED_URL_SCHEMES = {'http', 'https'}

# Internal/private network patterns to block (SSRF protection)
_SSRF_BLOCKED = re.compile(
    r'^(localhost|127\.|10\.|172\.(1[6-9]|2[0-9]|3[0-1])\.|192\.168\.|0\.0\.0\.0|::1)',
    re.IGNORECASE,
)

IMAGE_MEDIA_TYPES = {'image'}
VIDEO_MEDIA_TYPES = {'video', 'youtube', 'instagram'}


class DetectionService:
    """Unified entry point for all deepfake detection requests."""

    # ── File-based detection ──────────────────────────────────────────────────

    def analyse_file(self, file_path, media_type):
        """
        Analyse a locally stored file.

        Args:
            file_path:   str — absolute path
            media_type:  str — 'image' | 'video'

        Returns unified result dict.
        """
        logger.info("DetectionService.analyse_file: %s (type=%s)", os.path.basename(str(file_path)), media_type)

        try:
            if media_type == 'image':
                from .image_detector import detect_image
                return detect_image(file_path, media_type='image')
            elif media_type in VIDEO_MEDIA_TYPES:
                from .video_detector import detect_video
                return detect_video(file_path, media_type=media_type)
            else:
                return _error_result(f"Unknown media type: {media_type}")
        except Exception as exc:
            logger.error("DetectionService.analyse_file error: %s", exc)
            return _error_result(str(exc))

    # ── URL-based detection ───────────────────────────────────────────────────

    def analyse_youtube(self, url):
        """
        Download and analyse a YouTube video.

        Returns dict with 'download_error' key set if download fails,
        plus the full detection result if successful.
        """
        logger.info("DetectionService.analyse_youtube: %s", url)

        url_error = _validate_url(url, platform='youtube')
        if url_error:
            return _error_result(url_error, source_type='youtube', url=url)

        tmp_path = None
        try:
            tmp_path, title = _download_youtube(url)
            result = self.analyse_file(tmp_path, media_type='youtube')
            result['source_url']  = url
            result['source_title'] = title
            result['source_type'] = 'youtube'
            return result
        except _DownloadError as exc:
            logger.warning("YouTube download failed for %s: %s", url, exc)
            return _error_result(
                str(exc),
                source_type='youtube',
                url=url,
                is_download_error=True,
            )
        except Exception as exc:
            logger.error("YouTube analysis error for %s: %s", url, exc)
            return _error_result(str(exc), source_type='youtube', url=url)
        finally:
            _cleanup_temp(tmp_path)

    def analyse_instagram(self, url):
        """
        Download and analyse an Instagram Reel.

        Returns dict with 'download_error' key set if access fails.
        """
        logger.info("DetectionService.analyse_instagram: %s", url)

        url_error = _validate_url(url, platform='instagram')
        if url_error:
            return _error_result(url_error, source_type='instagram', url=url)

        tmp_path = None
        try:
            tmp_path, caption = _download_instagram(url)
            result = self.analyse_file(tmp_path, media_type='instagram')
            result['source_url']    = url
            result['source_title']  = caption
            result['source_type']   = 'instagram'
            return result
        except _DownloadError as exc:
            logger.warning("Instagram download failed for %s: %s", url, exc)
            return _error_result(
                str(exc),
                source_type='instagram',
                url=url,
                is_download_error=True,
            )
        except Exception as exc:
            logger.error("Instagram analysis error for %s: %s", url, exc)
            return _error_result(str(exc), source_type='instagram', url=url)
        finally:
            _cleanup_temp(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
#  URL validation (SSRF protection)
# ─────────────────────────────────────────────────────────────────────────────

def _validate_url(url, platform=None):
    """
    Validate a URL for safety and platform conformance.
    Returns error string or None if valid.
    """
    if not url or not isinstance(url, str):
        return "No URL provided."

    url = url.strip()

    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return "Invalid URL format."

    if parsed.scheme not in ALLOWED_URL_SCHEMES:
        return f"Invalid URL scheme '{parsed.scheme}'. Only http/https are allowed."

    hostname = parsed.hostname or ''
    if _SSRF_BLOCKED.match(hostname):
        return "URL points to an internal/private network address which is not allowed."

    if platform == 'youtube':
        if not any(d in hostname for d in ('youtube.com', 'youtu.be')):
            return "URL does not appear to be a YouTube URL. Please provide a YouTube link."

    if platform == 'instagram':
        if 'instagram.com' not in hostname:
            return "URL does not appear to be an Instagram URL. Please provide an Instagram Reel link."

    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Download helpers
# ─────────────────────────────────────────────────────────────────────────────

class _DownloadError(Exception):
    """Raised for user-facing download failures (private video, removed, etc.)"""
    pass


def _download_youtube(url):
    """Download YouTube video to a temp file. Returns (path, title)."""
    try:
        import yt_dlp as ydl_lib
    except ImportError:
        try:
            import youtube_dl as ydl_lib
        except ImportError:
            raise _DownloadError("YouTube download library (yt-dlp) is not installed.")

    tmp_dir = tempfile.mkdtemp(prefix='deepfake_yt_')
    ydl_opts = {
        'format':     'best[ext=mp4]/best',
        'outtmpl':    os.path.join(tmp_dir, '%(id)s.%(ext)s'),
        'quiet':      True,
        'no_warnings': True,
        'socket_timeout': 30,
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'ios', 'mweb']
            }
        }
    }

    # Try Chrome, Edge, Firefox, Brave, Safari, Opera for browser cookies extraction
    browsers = ['chrome', 'edge', 'firefox', 'brave', 'safari', 'opera']
    info = None
    video_path = None
    title = 'YouTube Video'
    last_exc = None

    for browser in browsers:
        try:
            opts = ydl_opts.copy()
            opts['cookiesfrombrowser'] = (browser, None, None, None)
            logger.info("Attempting YouTube download using cookies from %s", browser)
            with ydl_lib.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = ydl.prepare_filename(info)
                title = info.get('title', 'YouTube Video')
                break
        except Exception as exc:
            logger.warning("YouTube download failed using cookies from %s: %s", browser, exc)
            last_exc = exc
            continue

    if not video_path:
        # Fallback to downloading without browser cookies
        try:
            logger.info("Falling back to standard YouTube download without cookies")
            with ydl_lib.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = ydl.prepare_filename(info)
                title = info.get('title', 'YouTube Video')
        except Exception as exc:
            last_exc = last_exc or exc
            err_str = str(last_exc).lower()
            if 'private' in err_str:
                raise _DownloadError("This video is private and cannot be accessed.")
            elif 'unavailable' in err_str or 'not available' in err_str:
                raise _DownloadError("This video is unavailable or has been removed.")
            elif 'age' in err_str:
                raise _DownloadError("This video is age-restricted and cannot be downloaded without authentication.")
            elif 'network' in err_str or 'connection' in err_str or 'timeout' in err_str:
                raise _DownloadError("Network error while downloading the video. Please try again.")
            else:
                raise _DownloadError(f"Could not download YouTube video: {last_exc}")

    if not os.path.exists(video_path):
        # Some formats change extension
        for f in os.listdir(tmp_dir):
            candidate = os.path.join(tmp_dir, f)
            if os.path.isfile(candidate):
                video_path = candidate
                break
        else:
            raise _DownloadError("Download completed but video file was not found.")

    return video_path, title


def _download_instagram(url):
    """Download Instagram Reel to a temp file. Returns (path, caption)."""
    # Try using instaloader first
    original_cwd = os.getcwd()
    try:
        import instaloader
        tmp_dir = tempfile.mkdtemp(prefix='deepfake_ig_')
        loader = instaloader.Instaloader(
            download_videos=True,
            download_pictures=False,
            download_comments=False,
            save_metadata=False,
            post_metadata_txt_pattern='',
            quiet=True,
        )
        os.chdir(tmp_dir)
        post = instaloader.Post.from_url(url)
        loader.download_post(post, target=f'reel_{post.shortcode}')

        video_files = []
        for root, _, files in os.walk(tmp_dir):
            for fn in files:
                if fn.lower().endswith('.mp4'):
                    video_files.append(os.path.join(root, fn))

        if video_files:
            caption = (post.caption or 'Instagram Reel')[:200]
            os.chdir(original_cwd)
            return video_files[0], caption
    except Exception as exc:
        logger.warning("Instaloader failed for Instagram download: %s. Trying yt-dlp fallback...", exc)
    finally:
        os.chdir(original_cwd)

    # Fallback to yt-dlp
    try:
        import yt_dlp as ydl_lib
    except ImportError:
        try:
            import youtube_dl as ydl_lib
        except ImportError:
            raise _DownloadError("Instagram download fallback library (yt-dlp) is not installed.")

    tmp_dir = tempfile.mkdtemp(prefix='deepfake_ig_ytdl_')
    ydl_opts = {
        'format':     'best[ext=mp4]/best',
        'outtmpl':    os.path.join(tmp_dir, '%(id)s.%(ext)s'),
        'quiet':      True,
        'no_warnings': True,
        'socket_timeout': 30,
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'ios', 'mweb']
            }
        }
    }

    browsers = ['chrome', 'edge', 'firefox', 'brave', 'safari', 'opera']
    info = None
    video_path = None
    title = 'Instagram Reel'
    last_exc = None

    for browser in browsers:
        try:
            opts = ydl_opts.copy()
            opts['cookiesfrombrowser'] = (browser, None, None, None)
            logger.info("Attempting Instagram download using cookies from %s", browser)
            with ydl_lib.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = ydl.prepare_filename(info)
                title = info.get('title', 'Instagram Reel')
                break
        except Exception as exc:
            logger.warning("Instagram download failed using cookies from %s: %s", browser, exc)
            last_exc = exc
            continue

    if not video_path:
        try:
            logger.info("Falling back to standard Instagram download without cookies")
            with ydl_lib.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = ydl.prepare_filename(info)
                title = info.get('title', 'Instagram Reel')
        except Exception as exc:
            last_exc = last_exc or exc
            err_str = str(last_exc).lower()
            if 'login' in err_str or 'private' in err_str:
                raise _DownloadError("Unable to access this Instagram Reel. It may be private or requires login.")
            else:
                raise _DownloadError(f"Instagram download failed: {last_exc}")

    if not os.path.exists(video_path):
        for f in os.listdir(tmp_dir):
            candidate = os.path.join(tmp_dir, f)
            if os.path.isfile(candidate):
                video_path = candidate
                break
        else:
            raise _DownloadError("Instagram download succeeded but media file not found.")

    return video_path, title



# ─────────────────────────────────────────────────────────────────────────────
#  Cleanup
# ─────────────────────────────────────────────────────────────────────────────

def _cleanup_temp(path):
    """Safely delete a temporary file or directory."""
    if not path:
        return
    try:
        import shutil
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        # Also clean up parent temp dir if it's ours
        parent = os.path.dirname(path)
        if os.path.isdir(parent) and ('deepfake_yt_' in parent or 'deepfake_ig_' in parent):
            shutil.rmtree(parent, ignore_errors=True)
    except Exception as exc:
        logger.debug("Temp cleanup warning: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
#  Error result helper
# ─────────────────────────────────────────────────────────────────────────────

def _error_result(message, source_type='upload', url=None, is_download_error=False):
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
        'explanation':        message,
        'signal_breakdown':   {},
        'is_uncertain':       True,
        'frames_analyzed':    0,
        'frames_with_face':   0,
        'suspicious_frames':  0,
        'fake_frame_pct':     0.0,
        'max_frame_score':    0.0,
        'temporal_consistency': 0.0,
        'processing_time':    0.0,
        'error':              message,
        'source_type':        source_type,
        'source_url':         url,
        'is_download_error':  is_download_error,
    }
