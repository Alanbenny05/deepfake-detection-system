from django.urls import path
from . import views

app_name = 'reports'

urlpatterns = [
    path('', views.reports_list, name='reports_list'),
    path('download/<int:report_id>/', views.download_report, name='download_report'),
]