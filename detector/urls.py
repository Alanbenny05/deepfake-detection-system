from django.urls import path
from . import views

app_name = 'detector'

urlpatterns = [
    path('upload/', views.upload_media, name='upload'),
    path('youtube/', views.youtube_detection, name='youtube_detection'),
    path('instagram/', views.instagram_detection, name='instagram_detection'),
    path('result/<int:detection_id>/', views.result_view, name='result'),
    path('report/<int:detection_id>/', views.generate_report, name='generate_report'),
    path('history/', views.detection_history, name='detection_history'),
]