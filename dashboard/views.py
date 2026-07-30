from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from detector.models import Detection
from datetime import datetime, timedelta
import json

@login_required
def dashboard(request):
    # Get all detections for user
    detections = Detection.objects.filter(media__user=request.user)
    
    # Statistics
    total_uploads = detections.count()
    real_count = detections.filter(result='real').count()
    fake_count = detections.filter(result='fake').count()
    
    # Calculate accuracy (percentage of detections with confidence > 70)
    high_confidence = detections.filter(confidence__gt=70).count()
    accuracy = (high_confidence / total_uploads * 100) if total_uploads > 0 else 0
    
    # Recent detections
    recent_detections = detections.order_by('-processed_at')[:10]
    
    # Chart data (last 7 days)
    dates = []
    real_data = []
    fake_data = []
    
    for i in range(7, -1, -1):
        date = datetime.now().date() - timedelta(days=i)
        dates.append(date.strftime('%b %d'))
        
        day_detections = detections.filter(processed_at__date=date)
        real_data.append(day_detections.filter(result='real').count())
        fake_data.append(day_detections.filter(result='fake').count())
    
    context = {
        'total_uploads': total_uploads,
        'real_count': real_count,
        'fake_count': fake_count,
        'accuracy': round(accuracy, 1),
        'recent_detections': recent_detections,
        'chart_labels': json.dumps(dates),
        'chart_real_data': json.dumps(real_data),
        'chart_fake_data': json.dumps(fake_data),
    }
    
    return render(request, 'dashboard.html', context)