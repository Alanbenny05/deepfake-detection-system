import os
import cv2
import tempfile
import youtube_dl
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from PIL import Image as PILImage
from django.conf import settings
from datetime import datetime
import requests
import instaloader

def extract_frames_from_video(video_path, max_frames=30):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if not cap.isOpened():
        return frames
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // max_frames)
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % frame_interval == 0 and len(frames) < max_frames:
            frames.append(frame)
        
        count += 1
    
    cap.release()
    return frames

def process_youtube_video(url):
    """Download and process YouTube video"""
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True
    }
    
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = ydl.prepare_filename(info)
        title = info.get('title', 'YouTube Video')
    
    return video_path, title

def process_instagram_reel(url):
    """Download and process Instagram reel"""
    loader = instaloader.Instaloader(
        download_videos=True,
        download_pictures=False,
        download_comments=False,
        save_metadata=False,
        post_metadata_txt_pattern=""
    )
    
    temp_dir = tempfile.mkdtemp()
    original_cwd = os.getcwd()
    try:
        os.chdir(temp_dir)

        # Download reel
        post = instaloader.Post.from_url(url)
        loader.download_post(post, target=f"reel_{post.shortcode}")

        # Search recursively for the video file
        video_files = []
        for root, _, files in os.walk(temp_dir):
            for file_name in files:
                if file_name.lower().endswith('.mp4'):
                    video_files.append(os.path.join(root, file_name))

        if not video_files:
            raise ValueError("No video file found")

        video_path = video_files[0]
        caption = post.caption or "Instagram Reel"
        return video_path, caption
    finally:
        os.chdir(original_cwd)

def generate_pdf_report(detection):
    """Generate PDF report for detection"""
    # Create temp file
    report_path = os.path.join(
        settings.MEDIA_ROOT, 
        'reports',
        f'report_{detection.id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # Create PDF
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.blue,
        alignment=TA_CENTER
    )
    
    # Content
    story = []
    
    # Title
    story.append(Paragraph("Deepfake Detection Report", title_style))
    story.append(Spacer(1, 0.5*inch))
    
    # Date
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Result card
    result_text = "REAL" if detection.result == 'real' else "AI GENERATED"
    color = "green" if detection.result == 'real' else "red"
    story.append(Paragraph(f"Result: <font color={color}><b>{result_text}</b></font>", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    # Confidence
    story.append(Paragraph(f"Confidence Score: {detection.confidence:.2f}%", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Details table
    data = [
        ['Property', 'Value'],
        ['File Name', detection.media.file_name],
        ['Media Type', detection.media.media_type.capitalize()],
        ['Upload Date', detection.media.uploaded_at.strftime('%Y-%m-%d %H:%M')],
        ['Processing Date', detection.processed_at.strftime('%Y-%m-%d %H:%M')],
        ['Model Version', detection.model_version],
        ['Probability (Real)', f"{detection.probability_real:.2f}%"],
        ['Probability (Fake)', f"{detection.probability_fake:.2f}%"],
    ]
    
    table = Table(data, colWidths=[2*inch, 3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 0.5*inch))
    
    # Remarks
    remarks = "This media appears to be real and authentic." if detection.result == 'real' else "This media shows characteristics of AI-generated content."
    story.append(Paragraph("<b>Remarks:</b>", styles['Normal']))
    story.append(Paragraph(remarks, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Footer
    story.append(Paragraph("Report generated by AI-Based Deepfake Detection System", styles['Normal']))
    story.append(Paragraph("© 2024 All Rights Reserved", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    
    return report_path