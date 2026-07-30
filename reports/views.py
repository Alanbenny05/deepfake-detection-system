from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, Http404
from detector.models import Report
import os
from django.conf import settings

@login_required
def reports_list(request):
    reports = Report.objects.filter(
        detection__media__user=request.user
    ).order_by('-generated_at')
    
    return render(request, 'reports/list.html', {'reports': reports})

@login_required
def download_report(request, report_id):
    report = get_object_or_404(Report, id=report_id, detection__media__user=request.user)
    
    if not os.path.exists(report.pdf_path):
        raise Http404("Report file not found")
    
    with open(report.pdf_path, 'rb') as pdf:
        response = HttpResponse(pdf.read(), content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="deepfake_report_{report_id}.pdf"'
        return response