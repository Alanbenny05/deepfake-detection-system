from django.contrib import admin
from .models import Media, Detection, Report

@admin.register(Media)
class MediaAdmin(admin.ModelAdmin):
    list_display = ('file_name', 'media_type', 'user', 'file_size', 'uploaded_at')
    list_filter = ('media_type', 'uploaded_at')
    search_fields = ('file_name', 'user__username', 'url')
    date_hierarchy = 'uploaded_at'

@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    list_display = ('id', 'media', 'result', 'confidence', 'probability_real', 'probability_fake', 'processed_at')
    list_filter = ('result', 'model_version', 'processed_at')
    search_fields = ('media__file_name', 'media__user__username')
    readonly_fields = ('processed_at',)

@admin.register(Report)
class ReportAdmin(admin.ModelAdmin):
    list_display = ('id', 'detection', 'pdf_path', 'generated_at')
    list_filter = ('generated_at',)
    search_fields = ('detection__media__file_name',)
    readonly_fields = ('generated_at',)

