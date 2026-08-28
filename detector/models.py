from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


class Media(models.Model):
    MEDIA_TYPES = [
        ('image', 'Image'),
        ('video', 'Video'),
        ('youtube', 'YouTube'),
        ('instagram', 'Instagram'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    file_name = models.CharField(max_length=255)
    file_path = models.CharField(max_length=500, blank=True, null=True)
    media_type = models.CharField(max_length=20, choices=MEDIA_TYPES)
    file_size = models.IntegerField(default=0)
    url = models.URLField(blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    @property
    def media_url(self):
        if self.file_path and not self.file_path.startswith('{'):
            import os
            from django.conf import settings
            media_root = str(settings.MEDIA_ROOT)
            norm_path = os.path.normpath(self.file_path)
            norm_media = os.path.normpath(media_root)
            if norm_path.startswith(norm_media):
                rel_path = os.path.relpath(norm_path, norm_media).replace('\\', '/')
                return f"{settings.MEDIA_URL}{rel_path}"
            return self.file_path
        return self.url or ''

    def __str__(self):
        return f"{self.file_name} - {self.media_type}"


class Detection(models.Model):
    RESULT_CHOICES = [
        ('real',      'Real'),
        ('fake',      'AI Generated'),
        ('uncertain', 'Uncertain'),
    ]

    CLASSIFICATION_CHOICES = [
        ('HIGH_CONFIDENCE_REAL', 'High Confidence Real'),
        ('LIKELY_REAL',          'Likely Real'),
        ('UNCERTAIN',            'Uncertain'),
        ('LIKELY_FAKE',          'Likely Fake'),
        ('HIGH_CONFIDENCE_FAKE', 'High Confidence Fake'),
    ]

    # ── Existing fields (preserved exactly — no migration breakage) ────────────
    media            = models.OneToOneField(Media, on_delete=models.CASCADE, null=True, blank=True)
    result           = models.CharField(max_length=20, choices=RESULT_CHOICES, default='uncertain')
    confidence       = models.FloatField(default=0.0)
    probability_real = models.FloatField(default=0.0)
    probability_fake = models.FloatField(default=0.0)
    processed_at     = models.DateTimeField(auto_now_add=True)
    model_version    = models.CharField(max_length=50, default='v2.0')

    # ── New multi-signal fields (all nullable for backward compat) ─────────────
    cnn_score            = models.FloatField(null=True, blank=True,
                               help_text="Raw CNN fake probability [0-1]")
    forensic_score       = models.FloatField(null=True, blank=True,
                               help_text="Forensic analysis fake probability [0-1]")
    provenance_status    = models.CharField(max_length=20, blank=True, default='unavailable',
                               help_text="detected | not_detected | unavailable | error")
    provenance_markers   = models.TextField(blank=True, default='',
                               help_text="JSON list of found provenance markers")
    frames_analyzed      = models.IntegerField(null=True, blank=True,
                               help_text="Number of video frames analysed")
    suspicious_frames    = models.IntegerField(null=True, blank=True,
                               help_text="Number of frames flagged as suspicious")
    fake_frame_pct       = models.FloatField(null=True, blank=True,
                               help_text="Percentage of face frames that appear fake")
    temporal_consistency = models.FloatField(null=True, blank=True,
                               help_text="Temporal consistency score [0-1] for video")
    final_score          = models.FloatField(null=True, blank=True,
                               help_text="Ensemble final fake probability [0-1]")
    classification       = models.CharField(
                               max_length=25, blank=True, default='UNCERTAIN',
                               choices=CLASSIFICATION_CHOICES,
                               help_text="Ensemble classification tier")
    explanation          = models.TextField(blank=True, default='',
                               help_text="Human-readable explanation of the result")
    processing_time      = models.FloatField(null=True, blank=True,
                               help_text="Total detection processing time in seconds")
    source_type          = models.CharField(max_length=20, blank=True, default='upload',
                               help_text="upload | youtube | instagram")
    forensic_flags       = models.TextField(blank=True, default='',
                               help_text="JSON list of forensic warning flags")
    face_found           = models.BooleanField(null=True, blank=True,
                               help_text="Whether a face was detected in the media")
    cnn_mode             = models.CharField(max_length=20, blank=True, default='',
                               help_text="cnn | heuristic | unavailable")

    def __str__(self):
        name = self.media.file_name if self.media else 'No media'
        return f"{name} — {self.classification} ({self.confidence:.1f}%)"

    @property
    def is_real(self):
        return self.result == 'real'

    @property
    def is_fake(self):
        return self.result == 'fake'

    @property
    def is_uncertain(self):
        return self.result == 'uncertain'

    @property
    def classification_label(self):
        labels = {
            'HIGH_CONFIDENCE_REAL': 'High Confidence Real',
            'LIKELY_REAL':          'Likely Real',
            'UNCERTAIN':            'Uncertain',
            'LIKELY_FAKE':          'Likely Fake',
            'HIGH_CONFIDENCE_FAKE': 'High Confidence Fake',
        }
        return labels.get(self.classification, self.classification or 'Uncertain')

    @property
    def result_css_class(self):
        """Returns a CSS class name for styling the result badge."""
        mapping = {
            'HIGH_CONFIDENCE_REAL': 'result-high-real',
            'LIKELY_REAL':          'result-likely-real',
            'UNCERTAIN':            'result-uncertain',
            'LIKELY_FAKE':          'result-likely-fake',
            'HIGH_CONFIDENCE_FAKE': 'result-high-fake',
        }
        return mapping.get(self.classification, 'result-uncertain')


class Report(models.Model):
    detection    = models.OneToOneField(Detection, on_delete=models.CASCADE, null=True, blank=True)
    pdf_path     = models.CharField(max_length=500, blank=True, null=True)
    generated_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        name = (self.detection.media.file_name
                if self.detection and self.detection.media else 'No media')
        return f"Report for {name}"