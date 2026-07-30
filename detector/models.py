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
    
    def __str__(self):
        return f"{self.file_name} - {self.media_type}"

class Detection(models.Model):
    RESULT_CHOICES = [
        ('real', 'Real'),
        ('fake', 'AI Generated'),
    ]
    
    media = models.OneToOneField(Media, on_delete=models.CASCADE, null=True, blank=True)
    result = models.CharField(max_length=20, choices=RESULT_CHOICES, default='unknown')
    confidence = models.FloatField(default=0.0)
    probability_real = models.FloatField(default=0.0)
    probability_fake = models.FloatField(default=0.0)
    processed_at = models.DateTimeField(auto_now_add=True)
    model_version = models.CharField(max_length=50, default='v1.0')
    
    def __str__(self):
        return f"{self.media.file_name if self.media else 'No media'} - {self.result} ({self.confidence:.2f}%)"

class Report(models.Model):
    detection = models.OneToOneField(Detection, on_delete=models.CASCADE, null=True, blank=True)
    pdf_path = models.CharField(max_length=500, blank=True, null=True)
    generated_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Report for {self.detection.media.file_name if self.detection and self.detection.media else 'No media'}"