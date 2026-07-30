from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', TemplateView.as_view(template_name='index.html'), name='home'),
    path('users/', include('users.urls')),  # This already has app_name='users'
    path('detector/', include('detector.urls')),  # This already has app_name='detector'
    path('dashboard/', include('dashboard.urls')),  # This already has app_name='dashboard'
    path('history/', include('history.urls')),  # This already has app_name='history'
    path('reports/', include('reports.urls')),  # This already has app_name='reports'
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)