"""
Detector Views
==============
Thin Django views that delegate all detection logic to DetectionService.

No detection logic lives here — views only:
  1. Validate the HTTP request
  2. Save the uploaded file / URL to the database
  3. Call DetectionService
  4. Save the detection result to the database
  5. Update user profile stats
  6. Render the result template or redirect

IMPORTANT: There are NO random fallback values anywhere in this file.
If detection fails, the result is UNCERTAIN with a clear explanation.
"""

import json
import logging
import os
from datetime import datetime

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.http import FileResponse, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render

from .models import Detection, Media, Report
from .utils import generate_pdf_report

logger = logging.getLogger(__name__)

# File size limits
MAX_IMAGE_SIZE = 10 * 1024 * 1024    # 10 MB
MAX_VIDEO_SIZE = 100 * 1024 * 1024   # 100 MB

ALLOWED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}
ALLOWED_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: save detection result dict → Detection model
# ─────────────────────────────────────────────────────────────────────────────

def _save_detection(media, result_dict, source_type='upload'):
    """
    Persist a detection result to the database.

    Maps the unified result dict from DetectionService into Detection fields.
    Returns a Detection instance.
    """
    classification = result_dict.get('classification', 'UNCERTAIN')

    # Map classification → legacy result field (for backward compat)
    if classification in ('HIGH_CONFIDENCE_REAL', 'LIKELY_REAL'):
        legacy_result = 'real'
    elif classification in ('HIGH_CONFIDENCE_FAKE', 'LIKELY_FAKE'):
        legacy_result = 'fake'
    else:
        legacy_result = 'uncertain'

    final_score    = result_dict.get('final_score', 0.5)
    confidence     = result_dict.get('confidence', 0.0)
    prob_fake      = final_score * 100.0
    prob_real      = (1.0 - final_score) * 100.0

    # Serialize provenance markers list to string
    markers = result_dict.get('provenance_markers', [])
    markers_json = json.dumps(markers) if markers else ''

    # Serialize forensic flags
    flags = result_dict.get('forensic_flags', [])
    flags_json = json.dumps(flags) if flags else ''

    detection = Detection.objects.create(
        media             = media,
        result            = legacy_result,
        confidence        = round(confidence, 2),
        probability_real  = round(prob_real, 2),
        probability_fake  = round(prob_fake, 2),
        model_version     = 'v2.0',
        # New fields
        cnn_score         = result_dict.get('cnn_score'),
        forensic_score    = result_dict.get('forensic_score'),
        provenance_status = result_dict.get('provenance_status', 'unavailable'),
        provenance_markers= markers_json,
        frames_analyzed   = result_dict.get('frames_analyzed'),
        suspicious_frames = result_dict.get('suspicious_frames'),
        fake_frame_pct    = result_dict.get('fake_frame_pct'),
        temporal_consistency = result_dict.get('temporal_consistency'),
        final_score       = final_score,
        classification    = classification,
        explanation       = result_dict.get('explanation', ''),
        processing_time   = result_dict.get('processing_time'),
        source_type       = source_type,
        forensic_flags    = flags_json,
        face_found        = result_dict.get('face_found'),
        cnn_mode          = result_dict.get('cnn_mode', ''),
    )

    logger.info(
        "Detection saved: id=%d classification=%s confidence=%.1f%% media=%s",
        detection.id, classification, confidence, media.file_name
    )
    return detection


def _update_profile_stats(user, detection):
    """Update user profile detection counters."""
    try:
        profile = user.profile
        profile.total_detections += 1
        if detection.result == 'real':
            profile.real_detections += 1
        elif detection.result == 'fake':
            profile.fake_detections += 1
        profile.save()
    except Exception as exc:
        logger.warning("Could not update profile stats for %s: %s", user.username, exc)


def _validate_upload(media_file, media_type):
    """Validate an uploaded file. Returns error string or None."""
    if not media_file:
        return "Please select a file to upload."

    ext = os.path.splitext(media_file.name)[1].lower()

    if media_type == 'image':
        if ext not in ALLOWED_IMAGE_EXTS:
            return f"Unsupported image format '{ext}'. Please upload JPG, PNG, or WebP."
        if media_file.size > MAX_IMAGE_SIZE:
            return f"Image is too large ({media_file.size // 1024 // 1024}MB). Maximum is 10MB."
    elif media_type == 'video':
        if ext not in ALLOWED_VIDEO_EXTS:
            return f"Unsupported video format '{ext}'. Please upload MP4, MOV, or AVI."
        if media_file.size > MAX_VIDEO_SIZE:
            return f"Video is too large ({media_file.size // 1024 // 1024}MB). Maximum is 100MB."
    else:
        return "Invalid media type."

    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Views
# ─────────────────────────────────────────────────────────────────────────────

@login_required
def upload_media(request):
    if request.method == 'POST':
        media_type  = request.POST.get('media_type', '').strip()
        media_file  = request.FILES.get('media_file')

        # Validate
        error = _validate_upload(media_file, media_type)
        if error:
            messages.error(request, error)
            return render(request, 'detector/upload.html')

        # Deduplication check: if this user uploaded this exact file in the last 8s, redirect to existing result
        from django.utils import timezone
        import datetime
        recent_cutoff = timezone.now() - datetime.timedelta(seconds=8)
        recent_media = Media.objects.filter(
            user=request.user,
            file_name=media_file.name,
            file_size=media_file.size,
            uploaded_at__gte=recent_cutoff
        ).order_by('-uploaded_at').first()

        if recent_media and hasattr(recent_media, 'detection') and recent_media.detection:
            logger.info("Duplicate upload suppressed for %s (detection #%d)", media_file.name, recent_media.detection.id)
            return redirect('detector:result', detection_id=recent_media.detection.id)

        # Save file to media storage
        fs = FileSystemStorage()
        filename  = fs.save(f'uploads/{media_file.name}', media_file)
        file_path = fs.path(filename)

        # Create Media record
        media = Media.objects.create(
            user       = request.user,
            file_name  = media_file.name,
            file_path  = file_path,
            media_type = media_type,
            file_size  = media_file.size,
        )

        logger.info("Upload received: %s (%s) by user %s", media_file.name, media_type, request.user.username)

        # Run detection
        try:
            from .detection_service import DetectionService
            service = DetectionService()
            result = service.analyse_file(file_path, media_type=media_type)
        except Exception as exc:
            logger.error("Detection service error: %s", exc)
            result = {
                'status': 'error',
                'classification': 'UNCERTAIN',
                'label': 'Uncertain',
                'confidence': 0.0,
                'final_score': 0.5,
                'explanation': f'Detection service encountered an unexpected error: {exc}',
                'error': str(exc),
            }

        detection = _save_detection(media, result, source_type='upload')
        _update_profile_stats(request.user, detection)

        if result.get('error') and result.get('status') == 'error' and not result.get('is_download_error'):
            messages.warning(request, f"Analysis completed with issues: {result['error']}")
        else:
            messages.success(request, 'Analysis complete!')

        return redirect('detector:result', detection_id=detection.id)

    return render(request, 'detector/upload.html')


@login_required
def youtube_detection(request):
    if request.method == 'POST':
        url = request.POST.get('url', '').strip()

        if not url:
            messages.error(request, 'Please enter a YouTube URL.')
            return render(request, 'detector/upload.html')

        url_lower = url.lower()
        if 'instagram.com' in url_lower or ('youtube.com' not in url_lower and 'youtu.be' not in url_lower and 'youtube-nocookie.com' not in url_lower):
            messages.error(request, 'only use youtube link')
            return redirect('detector:upload')

        # Deduplication check
        from django.utils import timezone
        import datetime
        recent_cutoff = timezone.now() - datetime.timedelta(seconds=8)
        recent_media = Media.objects.filter(
            user=request.user,
            media_type='youtube',
            url=url,
            uploaded_at__gte=recent_cutoff
        ).order_by('-uploaded_at').first()

        if recent_media and hasattr(recent_media, 'detection') and recent_media.detection:
            logger.info("Duplicate YouTube submission suppressed for %s (detection #%d)", url, recent_media.detection.id)
            return redirect('detector:result', detection_id=recent_media.detection.id)

        logger.info("YouTube detection requested: %s by %s", url, request.user.username)

        # Create a placeholder Media record immediately
        media = Media.objects.create(
            user       = request.user,
            file_name  = 'YouTube Video',
            media_type = 'youtube',
            url        = url,
        )

        try:
            from .detection_service import DetectionService
            service = DetectionService()
            result  = service.analyse_youtube(url)
        except Exception as exc:
            logger.error("YouTube detection service error: %s", exc)
            result = {
                'status': 'error',
                'classification': 'UNCERTAIN',
                'label': 'Uncertain',
                'confidence': 0.0,
                'final_score': 0.5,
                'explanation': f'Could not analyse the YouTube video: {exc}',
                'error': str(exc),
                'is_download_error': True,
            }

        # Update file_name if title obtained
        title = result.get('source_title', '')
        if title and title != 'YouTube Video':
            media.file_name = title[:255]
            media.save(update_fields=['file_name'])

        if result.get('is_download_error'):
            messages.error(request, result.get('explanation') or result.get('error', 'YouTube download failed.'))
        else:
            messages.success(request, 'YouTube analysis complete!')

        # Update file size and dimensions metadata for YouTube media
        meta = result.get('video_metadata', {})
        if meta:
            w = meta.get('width', 0)
            h = meta.get('height', 0)
            fps = meta.get('fps', 0)
            dur = meta.get('duration_seconds', 0.0)
            dim_str = f"{w} x {h} px  |  {fps:.0f} fps  |  {dur:.1f}s"
            import json
            media.file_path = json.dumps({'dimensions': dim_str})
            media.file_size = meta.get('file_size', 0)
            media.save(update_fields=['file_path', 'file_size'])

        detection = _save_detection(media, result, source_type='youtube')
        _update_profile_stats(request.user, detection)

        return redirect('detector:result', detection_id=detection.id)

    return redirect('detector:upload')


@login_required
def instagram_detection(request):
    if request.method == 'POST':
        url = request.POST.get('url', '').strip()

        if not url:
            messages.error(request, 'Please enter an Instagram URL.')
            return render(request, 'detector/upload.html')

        url_lower = url.lower()
        if 'youtube.com' in url_lower or 'youtu.be' in url_lower or 'instagram.com' not in url_lower:
            messages.error(request, 'only use instagram link')
            return redirect('detector:upload')

        # Deduplication check
        from django.utils import timezone
        import datetime
        recent_cutoff = timezone.now() - datetime.timedelta(seconds=8)
        recent_media = Media.objects.filter(
            user=request.user,
            media_type='instagram',
            url=url,
            uploaded_at__gte=recent_cutoff
        ).order_by('-uploaded_at').first()

        if recent_media and hasattr(recent_media, 'detection') and recent_media.detection:
            logger.info("Duplicate Instagram submission suppressed for %s (detection #%d)", url, recent_media.detection.id)
            return redirect('detector:result', detection_id=recent_media.detection.id)

        logger.info("Instagram detection requested: %s by %s", url, request.user.username)

        media = Media.objects.create(
            user       = request.user,
            file_name  = 'Instagram Reel',
            media_type = 'instagram',
            url        = url,
        )

        try:
            from .detection_service import DetectionService
            service = DetectionService()
            result  = service.analyse_instagram(url)
        except Exception as exc:
            logger.error("Instagram detection service error: %s", exc)
            result = {
                'status': 'error',
                'classification': 'UNCERTAIN',
                'label': 'Uncertain',
                'confidence': 0.0,
                'final_score': 0.5,
                'explanation': f'Could not analyse the Instagram Reel: {exc}',
                'error': str(exc),
                'is_download_error': True,
            }

        # Update file_name if caption obtained
        caption = result.get('source_title', '')
        if caption and caption != 'Instagram Reel':
            media.file_name = caption[:255]
            media.save(update_fields=['file_name'])

        if result.get('is_download_error'):
            messages.error(request, result.get('explanation') or result.get('error', 'Instagram download failed.'))
        else:
            messages.success(request, 'Instagram analysis complete!')

        # Update file size and dimensions metadata for Instagram media
        meta = result.get('video_metadata', {})
        if meta:
            w = meta.get('width', 0)
            h = meta.get('height', 0)
            fps = meta.get('fps', 0)
            dur = meta.get('duration_seconds', 0.0)
            dim_str = f"{w} x {h} px  |  {fps:.0f} fps  |  {dur:.1f}s"
            import json
            media.file_path = json.dumps({'dimensions': dim_str})
            media.file_size = meta.get('file_size', 0)
            media.save(update_fields=['file_path', 'file_size'])

        detection = _save_detection(media, result, source_type='instagram')
        _update_profile_stats(request.user, detection)

        return redirect('detector:result', detection_id=detection.id)

    return redirect('detector:upload')


@login_required
def result_view(request, detection_id):
    try:
        detection = Detection.objects.get(id=detection_id, media__user=request.user)
        media     = detection.media

        # Parse stored JSON fields back to lists
        try:
            provenance_markers = json.loads(detection.provenance_markers) if detection.provenance_markers else []
        except (json.JSONDecodeError, TypeError):
            provenance_markers = []

        try:
            forensic_flags = json.loads(detection.forensic_flags) if detection.forensic_flags else []
        except (json.JSONDecodeError, TypeError):
            forensic_flags = []

        # Chart data for the probability pie
        chart_data = {
            'labels': ['Real', 'AI Generated'],
            'data':   [detection.probability_real, detection.probability_fake],
            'colors': ['#22c55e', '#ef4444'],
        }

        # Build signal breakdown for the multi-signal UI
        signal_data = {
            'cnn': {
                'score':     detection.cnn_score,
                'mode':      detection.cnn_mode or 'unknown',
                'available': detection.cnn_score is not None,
            },
            'forensic': {
                'score':     detection.forensic_score,
                'flags':     forensic_flags,
                'available': detection.forensic_score is not None,
            },
            'provenance': {
                'status':    detection.provenance_status or 'unavailable',
                'markers':   provenance_markers,
                'available': detection.provenance_status not in ('unavailable', 'error', ''),
            },
            'video': {
                'frames_analyzed':   detection.frames_analyzed,
                'suspicious_frames': detection.suspicious_frames,
                'fake_frame_pct':    detection.fake_frame_pct,
                'temporal_consistency': detection.temporal_consistency,
                'available': detection.frames_analyzed is not None,
            },
        }

        context = {
            'detection':          detection,
            'media':              media,
            'chart_data':         json.dumps(chart_data),
            'signal_data':        json.dumps(signal_data),
            'signal_data_py':     signal_data,
            'provenance_markers': provenance_markers,
            'forensic_flags':     forensic_flags,
            'is_real':            detection.result == 'real',
            'is_fake':            detection.result == 'fake',
            'is_uncertain':       detection.result == 'uncertain',
            'classification':     detection.classification or 'UNCERTAIN',
            'classification_label': detection.classification_label,
        }

        return render(request, 'detector/result.html', context)

    except Detection.DoesNotExist:
        messages.error(request, 'Detection not found.')
        return redirect('detector:upload')


@login_required
def generate_report(request, detection_id):
    detection   = get_object_or_404(Detection, id=detection_id, media__user=request.user)
    report_path = generate_pdf_report(detection)

    report, _ = Report.objects.get_or_create(detection=detection)
    report.pdf_path = report_path
    report.save()

    if not os.path.exists(report_path):
        messages.error(request, 'Unable to generate report file.')
        return redirect('detector:result', detection_id=detection_id)

    dt_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'deepfake_report_{dt_str}.pdf'

    response = FileResponse(open(report_path, 'rb'), content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response


@login_required
def detection_history(request):
    detections = Detection.objects.filter(
        media__user=request.user
    ).order_by('-processed_at').select_related('media')

    return render(request, 'detector/history.html', {'detections': detections})