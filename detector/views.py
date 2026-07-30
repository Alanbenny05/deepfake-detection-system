from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponse
from .models import Media, Detection, Report
import json

@login_required
def upload_media(request):
    if request.method == 'POST':
        media_type = request.POST.get('media_type')
        media_file = request.FILES.get('media_file')
        
        if not media_file:
            messages.error(request, 'Please select a file to upload.')
            return render(request, 'detector/upload.html')
        
        # Create media record
        media = Media.objects.create(
            user=request.user,
            file_name=media_file.name,
            media_type=media_type,
            file_size=media_file.size
        )
        
        # For demonstration, create a random detection result
        import random
        result = random.choice(['real', 'fake'])
        confidence = random.uniform(70, 98)
        prob_real = confidence if result == 'real' else 100 - confidence
        prob_fake = 100 - confidence if result == 'real' else confidence
        
        detection = Detection.objects.create(
            media=media,
            result=result,
            confidence=confidence,
            probability_real=prob_real,
            probability_fake=prob_fake
        )
        
        # Update profile stats
        profile = request.user.profile
        profile.total_detections += 1
        if result == 'real':
            profile.real_detections += 1
        else:
            profile.fake_detections += 1
        profile.save()
        
        messages.success(request, 'Analysis complete!')
        return redirect('detector:result', detection_id=detection.id)
    
    return render(request, 'detector/upload.html')

@login_required
def youtube_detection(request):
    if request.method == 'POST':
        url = request.POST.get('url')
        
        if not url:
            messages.error(request, 'Please enter a YouTube URL.')
            return render(request, 'detector/upload.html')
        
        # Create media record
        media = Media.objects.create(
            user=request.user,
            file_name=f"YouTube Video - {url[:30]}",
            media_type='youtube',
            url=url
        )
        
        # For demonstration, create random detection
        import random
        result = random.choice(['real', 'fake'])
        confidence = random.uniform(70, 98)
        prob_real = confidence if result == 'real' else 100 - confidence
        prob_fake = 100 - confidence if result == 'real' else confidence
        
        detection = Detection.objects.create(
            media=media,
            result=result,
            confidence=confidence,
            probability_real=prob_real,
            probability_fake=prob_fake
        )
        
        # Update profile stats
        profile = request.user.profile
        profile.total_detections += 1
        if result == 'real':
            profile.real_detections += 1
        else:
            profile.fake_detections += 1
        profile.save()
        
        messages.success(request, 'YouTube analysis complete!')
        return redirect('detector:result', detection_id=detection.id)
    
    return redirect('detector:upload')

@login_required
def instagram_detection(request):
    if request.method == 'POST':
        url = request.POST.get('url')
        
        if not url:
            messages.error(request, 'Please enter an Instagram URL.')
            return render(request, 'detector/upload.html')
        
        # Create media record
        media = Media.objects.create(
            user=request.user,
            file_name=f"Instagram Reel - {url[:30]}",
            media_type='instagram',
            url=url
        )
        
        # For demonstration, create random detection
        import random
        result = random.choice(['real', 'fake'])
        confidence = random.uniform(70, 98)
        prob_real = confidence if result == 'real' else 100 - confidence
        prob_fake = 100 - confidence if result == 'real' else confidence
        
        detection = Detection.objects.create(
            media=media,
            result=result,
            confidence=confidence,
            probability_real=prob_real,
            probability_fake=prob_fake
        )
        
        # Update profile stats
        profile = request.user.profile
        profile.total_detections += 1
        if result == 'real':
            profile.real_detections += 1
        else:
            profile.fake_detections += 1
        profile.save()
        
        messages.success(request, 'Instagram analysis complete!')
        return redirect('detector:result', detection_id=detection.id)
    
    return redirect('detector:upload')

@login_required
def result_view(request, detection_id):
    try:
        detection = Detection.objects.get(id=detection_id, media__user=request.user)
        media = detection.media
        
        chart_data = {
            'labels': ['Real', 'AI Generated'],
            'data': [detection.probability_real, detection.probability_fake],
            'colors': ['#16A34A', '#DC2626']
        }
        
        context = {
            'detection': detection,
            'media': media,
            'chart_data': json.dumps(chart_data),
            'is_real': detection.result == 'real'
        }
        
        return render(request, 'detector/result.html', context)
    except Detection.DoesNotExist:
        messages.error(request, 'Detection not found.')
        return redirect('detector:upload')

@login_required
def generate_report(request, detection_id):
    # Simple PDF placeholder
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="report_{detection_id}.pdf"'
    response.write(b'PDF Report Placeholder - Deepfake Detection Results')
    return response

@login_required
def detection_history(request):
    detections = Detection.objects.filter(
        media__user=request.user
    ).order_by('-processed_at')
    
    return render(request, 'detector/history.html', {'detections': detections})