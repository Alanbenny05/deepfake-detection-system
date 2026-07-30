from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from detector.models import Detection

@login_required
def history_view(request):
    detections = Detection.objects.filter(
        media__user=request.user
    ).order_by('-processed_at')
    
    return render(request, 'detector/history.html', {'detections': detections})