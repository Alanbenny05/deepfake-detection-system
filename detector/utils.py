import os
import io
import tempfile
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect, String, Circle
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderPDF
from PIL import Image as PILImage
from django.conf import settings


# ─────────────────────────────────────────────────────────────
#  Colour palette — matches system CSS variables exactly
#  (see static/css/styles.css :root)
# ─────────────────────────────────────────────────────────────

REAL_COLOR   = colors.HexColor('#22c55e')   # --success-color
FAKE_COLOR   = colors.HexColor('#ef4444')   # --danger-color
ACCENT_COLOR = colors.HexColor('#38bdf8')   # --accent-color  (sky blue)
PRIMARY      = colors.HexColor('#5b45d3')   # --primary-color (purple)
SECONDARY    = colors.HexColor('#7c3aed')   # --secondary-color
NAV_TOP      = colors.HexColor('#0f172a')   # navbar gradient top
NAV_BOT      = colors.HexColor('#111a47')   # navbar gradient bottom / --dark-panel
DARK_SURFACE = colors.HexColor('#15235e')   # --dark-surface
LIGHT_BG     = colors.HexColor('#eef2ff')   # body background
SURFACE      = colors.HexColor('#f8fbff')   # --surface
BORDER_COLOR = colors.HexColor('#cbd5e1')
TEXT_DARK    = colors.HexColor('#0f172a')   # --text-dark
TEXT_MUTED   = colors.HexColor('#64748b')   # --text-muted


# ─────────────────────────────────────────────────────────────
#  Style helpers
# ─────────────────────────────────────────────────────────────

def _styles():
    base = getSampleStyleSheet()
    custom = {}

    custom['ReportTitle'] = ParagraphStyle(
        'ReportTitle',
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=colors.white,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    custom['ReportSubtitle'] = ParagraphStyle(
        'ReportSubtitle',
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#c7d2fe'),
        alignment=TA_CENTER,
        spaceAfter=0,
    )
    custom['SectionHeader'] = ParagraphStyle(
        'SectionHeader',
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=PRIMARY,
        spaceBefore=10,
        spaceAfter=4,
        borderPad=4,
    )
    custom['BodyText'] = ParagraphStyle(
        'BodyText',
        fontName='Helvetica',
        fontSize=9,
        textColor=TEXT_DARK,
        spaceAfter=3,
        leading=14,
    )
    custom['BodyMuted'] = ParagraphStyle(
        'BodyMuted',
        fontName='Helvetica',
        fontSize=8,
        textColor=TEXT_MUTED,
        spaceAfter=2,
    )
    custom['ResultBadge'] = ParagraphStyle(
        'ResultBadge',
        fontName='Helvetica-Bold',
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    custom['ConfidenceLabel'] = ParagraphStyle(
        'ConfidenceLabel',
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=TEXT_DARK,
        alignment=TA_CENTER,
    )
    custom['SmallLabel'] = ParagraphStyle(
        'SmallLabel',
        fontName='Helvetica',
        fontSize=7,
        textColor=TEXT_MUTED,
        alignment=TA_CENTER,
    )
    return custom


# ─────────────────────────────────────────────────────────────
#  Confidence bar drawing (visual gauge)
# ─────────────────────────────────────────────────────────────

def _confidence_bar_drawing(prob_real, prob_fake, width=400):
    """Return a ReportLab Drawing showing a horizontal confidence bar."""
    bar_h = 20
    d = Drawing(width, bar_h + 30)

    total = prob_real + prob_fake
    if total == 0:
        total = 1
    real_w = int((prob_real / total) * width)
    fake_w = width - real_w

    # Real segment
    d.add(Rect(0, 10, real_w, bar_h, fillColor=REAL_COLOR, strokeColor=None))
    # Fake segment
    d.add(Rect(real_w, 10, fake_w, bar_h, fillColor=FAKE_COLOR, strokeColor=None))

    # Labels
    if real_w > 40:
        d.add(String(real_w / 2, 16, f'REAL {prob_real:.1f}%',
                     fontName='Helvetica-Bold', fontSize=7,
                     fillColor=colors.white, textAnchor='middle'))
    if fake_w > 40:
        d.add(String(real_w + fake_w / 2, 16, f'FAKE {prob_fake:.1f}%',
                     fontName='Helvetica-Bold', fontSize=7,
                     fillColor=colors.white, textAnchor='middle'))

    # Bottom labels
    d.add(String(0, 2, f'Real: {prob_real:.2f}%',
                 fontName='Helvetica', fontSize=6, fillColor=REAL_COLOR))
    d.add(String(width, 2, f'Fake: {prob_fake:.2f}%',
                 fontName='Helvetica', fontSize=6, fillColor=FAKE_COLOR,
                 textAnchor='end'))

    return d


def _score_gauge_drawing(confidence, is_fake, width=160, height=80):
    """Circular score indicator."""
    d = Drawing(width, height)
    cx, cy, r = width / 2, height / 2 + 5, 32
    ring_color = FAKE_COLOR if is_fake else REAL_COLOR

    # Outer ring
    d.add(Circle(cx, cy, r, fillColor=colors.white,
                 strokeColor=ring_color, strokeWidth=5))
    # Inner score text
    d.add(String(cx, cy, f'{confidence:.0f}%',
                 fontName='Helvetica-Bold', fontSize=16,
                 fillColor=ring_color, textAnchor='middle'))
    d.add(String(cx, cy - 14, 'Confidence',
                 fontName='Helvetica', fontSize=6,
                 fillColor=TEXT_MUTED, textAnchor='middle'))
    return d


# ─────────────────────────────────────────────────────────────
#  Image thumbnail helper
# ─────────────────────────────────────────────────────────────

def _get_image_thumbnail(media, max_w=3.5*inch, max_h=2.8*inch):
    """
    Try to get the uploaded image/video thumbnail as a ReportLab Image.
    Returns None if not available.
    """
    file_path = getattr(media, 'file_path', None)
    if not file_path or not os.path.exists(str(file_path)):
        return None

    media_type = getattr(media, 'media_type', '')

    try:
        if media_type in ('image',):
            # Directly use image
            pil_img = PILImage.open(file_path).convert('RGB')

        elif media_type in ('video', 'youtube', 'instagram'):
            # Extract first frame from video
            try:
                import cv2
                cap = cv2.VideoCapture(str(file_path))
                # Skip to 10% through video for a more meaningful frame
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 10))
                ret, frame = cap.read()
                cap.release()
                if not ret or frame is None:
                    return None
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = PILImage.fromarray(frame_rgb)
            except Exception:
                return None
        else:
            return None

        # Maintain aspect ratio and fit within max dimensions
        orig_w, orig_h = pil_img.size
        scale = min(max_w / orig_w, max_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pil_img = pil_img.resize((new_w, new_h), PILImage.LANCZOS)

        # Save to in-memory buffer
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=88)
        buf.seek(0)

        return RLImage(buf, width=new_w, height=new_h)

    except Exception:
        return None


def _get_file_size_str(media):
    size = getattr(media, 'file_size', 0) or 0
    if size == 0:
        # Try to get from file
        fp = getattr(media, 'file_path', None)
        if fp and os.path.exists(str(fp)):
            size = os.path.getsize(str(fp))
    if size >= 1024 * 1024:
        return f'{size / (1024 * 1024):.2f} MB'
    elif size >= 1024:
        return f'{size / 1024:.1f} KB'
    return f'{size} B'


def _get_image_dimensions(media):
    fp = getattr(media, 'file_path', None)
    media_type = getattr(media, 'media_type', '')
    if media_type in ('youtube', 'instagram') and fp:
        if str(fp).startswith('{'):
            try:
                import json
                data = json.loads(str(fp))
                return data.get('dimensions', 'N/A')
            except Exception:
                pass
        return str(fp)

    if not fp or not os.path.exists(str(fp)):
        return 'N/A'
    try:
        if media_type == 'image':
            with PILImage.open(str(fp)) as img:
                return f'{img.width} x {img.height} px'
        elif media_type in ('video', 'youtube', 'instagram'):
            import cv2
            cap = cv2.VideoCapture(str(fp))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frames / fps if fps > 0 else 0
            cap.release()
            return f'{w} x {h} px  |  {fps:.0f} fps  |  {duration:.1f}s'
    except Exception:
        pass
    return 'N/A'


# ─────────────────────────────────────────────────────────────
#  Main PDF generator
# ─────────────────────────────────────────────────────────────

def generate_pdf_report(detection):
    """Generate an enhanced, detailed PDF report for a deepfake detection result."""

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = os.path.join(
        settings.MEDIA_ROOT,
        'reports',
        f'deepfake_report_{timestamp}.pdf'
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    doc = SimpleDocTemplate(
        report_path,
        pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm
    )

    S = _styles()
    story = []
    page_w = A4[0] - 3.6*cm   # usable width

    classification = getattr(detection, 'classification', 'UNCERTAIN') or 'UNCERTAIN'
    is_uncertain = detection.result == 'uncertain' or classification == 'UNCERTAIN'
    is_fake = detection.result == 'fake' or 'FAKE' in classification

    if is_uncertain:
        result_color = colors.HexColor('#eab308')  # amber
        result_label = 'INCONCLUSIVE / UNCERTAIN'
    elif is_fake:
        result_color = FAKE_COLOR
        result_label = 'AI GENERATED / DEEPFAKE'
    else:
        result_color = REAL_COLOR
        result_label = 'AUTHENTIC / REAL'

    # ── 1. Header banner — matches system navbar exactly ────────
    gen_date = datetime.now().strftime('%B %d, %Y  at  %H:%M')

    brand_style = ParagraphStyle(
        'Brand',
        fontName='Helvetica-Bold',
        fontSize=17,
        textColor=colors.white,
        leading=22,
    )
    subtitle_style = ParagraphStyle(
        'NavSubtitle',
        fontName='Helvetica',
        fontSize=8.5,
        textColor=ACCENT_COLOR,
        leading=12,
    )
    logo_path = os.path.join(settings.BASE_DIR, 'static', 'img', 'robot_logo.png')
    if not os.path.exists(logo_path):
        logo_path = os.path.join(settings.BASE_DIR, 'static', 'img', 'robot_logo.jpg')
    logo_img = None
    if os.path.exists(logo_path):
        try:
            logo_img = RLImage(logo_path, width=28, height=28)
        except Exception:
            pass

    brand_para = Paragraph('Deepfake Detector', brand_style)
    subtitle_para = Paragraph('Multi-Signal Verification Report', subtitle_style)

    if logo_img:
        brand_subtable = Table(
            [[logo_img, [brand_para, subtitle_para]]],
            colWidths=[36, page_w * 0.70 - 36]
        )
        brand_subtable.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        brand_cell = brand_subtable
    else:
        brand_cell = [
            Paragraph(
                f'<font color="#38bdf8">[R]</font>  Deepfake Detector',
                brand_style
            ),
            Paragraph('Multi-Signal Verification Report', subtitle_style),
        ]

    date_style = ParagraphStyle(
        'NavDate',
        fontName='Helvetica',
        fontSize=8,
        textColor=colors.HexColor('#94a3b8'),
        alignment=TA_RIGHT,
    )
    date_cell = Paragraph(f'Generated<br/>{gen_date}', date_style)

    header_table = Table(
        [[brand_cell, date_cell]],
        colWidths=[page_w * 0.70, page_w * 0.30]
    )
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), NAV_TOP),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('TOPPADDING', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ('LEFTPADDING', (0, 0), (0, 0), 18),
        ('LEFTPADDING', (1, 0), (1, 0), 8),
        ('RIGHTPADDING', (0, 0), (0, 0), 8),
        ('RIGHTPADDING', (1, 0), (1, 0), 18),
        ('LINEBELOW', (0, 0), (-1, 0), 3, ACCENT_COLOR),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.28*inch))

    # ── 2. Result verdict + confidence gauge (side by side) ─────
    badge_hex = '#eab308' if is_uncertain else ('#ef4444' if is_fake else '#22c55e')
    verdict_para = Paragraph(
        f'<font color="{badge_hex}"><b>{result_label}</b></font><br/>'
        f'<font size="9" color="#64748b">Classification Tier: {detection.classification_label if hasattr(detection, "classification_label") else classification}</font>',
        S['ResultBadge']
    )
    gauge = _score_gauge_drawing(detection.confidence, is_fake)

    verdict_table = Table(
        [[verdict_para, gauge]],
        colWidths=[page_w * 0.65, page_w * 0.35]
    )
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_BG),
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
        ('ALIGN', (1, 0), (1, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ('LEFTPADDING', (0, 0), (-1, -1), 16),
        ('RIGHTPADDING', (0, 0), (-1, -1), 16),
        ('BOX', (0, 0), (-1, -1), 1.5, result_color),
        ('LINEAFTER', (0, 0), (0, -1), 0.5, BORDER_COLOR),
    ]))
    story.append(KeepTogether(verdict_table))
    story.append(Spacer(1, 0.2*inch))

    # ── 3. Multi-signal score breakdown ───────────────────────────
    story.append(Paragraph('Multi-Signal Evidence Analysis', S['SectionHeader']))
    story.append(HRFlowable(width=page_w, thickness=1, color=ACCENT_COLOR, spaceAfter=6))

    bar_drawing = _confidence_bar_drawing(
        detection.probability_real, detection.probability_fake,
        width=int(page_w * 0.9)
    )
    bar_wrapper = Table([[bar_drawing]], colWidths=[page_w])
    bar_wrapper.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
    ]))
    story.append(bar_wrapper)
    story.append(Spacer(1, 0.15*inch))

    # Signal breakdown table
    cnn_sc_str = f"{detection.cnn_score:.3f}" if getattr(detection, 'cnn_score', None) is not None else 'N/A'
    for_sc_str = f"{detection.forensic_score:.3f}" if getattr(detection, 'forensic_score', None) is not None else 'N/A'
    prov_str   = getattr(detection, 'provenance_status', 'unavailable') or 'unavailable'
    final_sc_str = f"{detection.final_score:.3f}" if getattr(detection, 'final_score', None) is not None else f"{detection.probability_fake/100.0:.3f}"

    score_rows = [
        ['Detection Signal', 'Signal Value', 'Status / Evaluation'],
        ['CNN Model Score', cnn_sc_str, f"Mode: {getattr(detection, 'cnn_mode', 'N/A') or 'N/A'}"],
        ['Forensic Analysis', for_sc_str, 'ELA, DCT, noise, sharpness'],
        ['Provenance / SynthID', prov_str.upper(), 'Metadata & credential markers'],
        ['Final Ensemble Score', final_sc_str, f"Tier: {classification}"],
        ['Overall Confidence', f'{detection.confidence:.1f}%', 'Probabilistic model estimate'],
    ]

    score_table = Table(score_rows, colWidths=[page_w*0.35, page_w*0.25, page_w*0.40])
    ts = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAV_BOT),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8.5),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('ALIGN', (1, 1), (1, -1), 'CENTER'),
        ('ALIGN', (2, 1), (2, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
    ])
    score_table.setStyle(ts)
    story.append(score_table)
    story.append(Spacer(1, 0.25*inch))

    # ── 4. Media preview ────────────────────────────────────────
    story.append(Paragraph('Media Preview & Attributes', S['SectionHeader']))
    story.append(HRFlowable(width=page_w, thickness=1, color=ACCENT_COLOR, spaceAfter=6))

    thumb = _get_image_thumbnail(detection.media) if detection.media else None
    dimensions = _get_image_dimensions(detection.media) if detection.media else 'N/A'
    file_size   = _get_file_size_str(detection.media) if detection.media else 'N/A'
    file_name   = detection.media.file_name if detection.media else 'N/A'
    media_type  = detection.media.media_type.capitalize() if detection.media else 'N/A'
    upload_date = detection.media.uploaded_at.strftime('%Y-%m-%d %H:%M') if detection.media else 'N/A'

    if thumb:
        meta_text = (
            f'<b>File:</b> {file_name}<br/>'
            f'<b>Type:</b> {media_type}<br/>'
            f'<b>Size:</b> {file_size}<br/>'
            f'<b>Dimensions:</b> {dimensions}<br/>'
            f'<b>Uploaded:</b> {upload_date}<br/>'
            f'<b>Analysis:</b> {result_label}'
        )
        meta_para = Paragraph(meta_text, S['BodyText'])
        badge_para = Paragraph(f'<font color="{badge_hex}"><b>[{result_label}]</b></font>', S['BodyMuted'])

        preview_table = Table(
            [[thumb, [meta_para, Spacer(1, 8), badge_para]]],
            colWidths=[page_w * 0.48, page_w * 0.52]
        )
        preview_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, 0), 'CENTER'),
            ('ALIGN', (1, 0), (1, 0), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('BOX', (0, 0), (-1, -1), 1, BORDER_COLOR),
            ('LINEAFTER', (0, 0), (0, -1), 0.5, BORDER_COLOR),
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ]))
    else:
        media_url_str = detection.media.url if detection.media else ''
        placeholder_text = f'<font color="#94a3b8">[No preview available for {media_type} media]</font>'
        if media_url_str:
            placeholder_text += (
                f'<br/><br/><b>Working Link:</b> '
                f'<font color="#2563eb"><u><a href="{media_url_str}">{media_url_str}</a></u></font>'
            )
        placeholder = Paragraph(placeholder_text, S['BodyText'])
        meta_text = (
            f'<b>File:</b> {file_name} &nbsp;&nbsp; '
            f'<b>Type:</b> {media_type} &nbsp;&nbsp; '
            f'<b>Size:</b> {file_size} &nbsp;&nbsp; '
            f'<b>Dimensions:</b> {dimensions} &nbsp;&nbsp; '
            f'<b>Uploaded:</b> {upload_date}'
        )
        preview_table = Table([[placeholder], [Paragraph(meta_text, S['BodyText'])]], colWidths=[page_w])
        preview_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ]))

    story.append(preview_table)
    story.append(Spacer(1, 0.25*inch))

    # ── 5. Interpretation & remarks ─────────────────────────────
    story.append(Paragraph('Analysis Interpretation & Explanation', S['SectionHeader']))
    story.append(HRFlowable(width=page_w, thickness=1, color=ACCENT_COLOR, spaceAfter=6))

    if getattr(detection, 'explanation', None):
        remarks = detection.explanation
    elif is_uncertain:
        remarks = (
            'The multi-signal detection pipeline returned an <b>uncertain</b> result. '
            'The available evidence from CNN analysis, forensic features, and provenance metadata was either '
            'inconclusive or contradictory. The system does not force a definitive Real or Fake classification '
            'when evidence is insufficient.'
        )
    elif is_fake:
        remarks = (
            'The multi-signal detection pipeline has classified this media as <b>AI-generated or manipulated</b>. '
            f'The system reports a <b>{detection.probability_fake:.1f}%</b> probability of synthetic generation. '
            'Characteristics consistent with deepfake artefacts or anomalies were identified.'
        )
    else:
        remarks = (
            'The multi-signal detection pipeline has classified this media as <b>authentic and real</b>. '
            f'The system reports a <b>{detection.probability_real:.1f}%</b> probability of authenticity. '
            'No significant deepfake artefacts were detected across the available signals.'
        )

    story.append(Paragraph(remarks, S['BodyText']))
    story.append(Spacer(1, 0.2*inch))

    # ── 6. Disclaimer ───────────────────────────────────────────
    disclaimer = (
        '<i>Disclaimer: This report is generated automatically by an AI-based multi-signal deepfake detection '
        'system and is provided for informational purposes only. Results are probabilistic and should not be used as '
        'sole evidence in legal proceedings without expert human review.</i>'
    )
    disc_table = Table([[Paragraph(disclaimer, S['BodyMuted'])]], colWidths=[page_w])
    disc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fef9c3')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#fde047')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(disc_table)

    doc.build(story)
    return report_path


# ─────────────────────────────────────────────────────────────
#  Other utilities (unchanged)
# ─────────────────────────────────────────────────────────────

def extract_frames_from_video(video_path, max_frames=30):
    """Extract frames from video."""
    try:
        import cv2
    except ImportError:
        return []

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
    """Download and process YouTube video using yt-dlp or youtube-dl."""
    try:
        import yt_dlp as ydl_lib
    except ImportError:
        try:
            import youtube_dl as ydl_lib
        except ImportError:
            raise ImportError("Neither yt-dlp nor youtube-dl is installed.")

    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'socket_timeout': 30,
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'ios', 'mweb']
            }
        }
    }

    browsers = ['chrome', 'edge', 'firefox', 'brave', 'safari', 'opera']
    info = None
    video_path = None
    title = 'YouTube Video'
    last_exc = None

    for browser in browsers:
        try:
            opts = ydl_opts.copy()
            opts['cookiesfrombrowser'] = (browser, None, None, None)
            with ydl_lib.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = ydl.prepare_filename(info)
                title = info.get('title', 'YouTube Video')
                return video_path, title
        except Exception as exc:
            last_exc = exc
            continue

    # Fallback to no cookies
    try:
        with ydl_lib.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
            title = info.get('title', 'YouTube Video')
            return video_path, title
    except Exception as exc:
        raise last_exc or exc


def process_instagram_reel(url):
    """Download and process Instagram reel."""
    # Try using instaloader first
    original_cwd = os.getcwd()
    try:
        import instaloader
        loader = instaloader.Instaloader(
            download_videos=True,
            download_pictures=False,
            download_comments=False,
            save_metadata=False,
            post_metadata_txt_pattern=""
        )
        temp_dir = tempfile.mkdtemp()
        os.chdir(temp_dir)
        post = instaloader.Post.from_url(url)
        loader.download_post(post, target=f"reel_{post.shortcode}")

        video_files = []
        for root, _, files in os.walk(temp_dir):
            for file_name in files:
                if file_name.lower().endswith('.mp4'):
                    video_files.append(os.path.join(root, file_name))

        if video_files:
            video_path = video_files[0]
            caption = post.caption or "Instagram Reel"
            os.chdir(original_cwd)
            return video_path, caption
    except Exception as exc:
        pass
    finally:
        os.chdir(original_cwd)

    # Fallback to yt-dlp
    try:
        import yt_dlp as ydl_lib
    except ImportError:
        try:
            import youtube_dl as ydl_lib
        except ImportError:
            raise ImportError("Neither yt-dlp nor youtube-dl is installed.")

    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'socket_timeout': 30,
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'ios', 'mweb']
            }
        }
    }

    browsers = ['chrome', 'edge', 'firefox', 'brave', 'safari', 'opera']
    info = None
    video_path = None
    title = 'Instagram Reel'
    last_exc = None

    for browser in browsers:
        try:
            opts = ydl_opts.copy()
            opts['cookiesfrombrowser'] = (browser, None, None, None)
            with ydl_lib.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = ydl.prepare_filename(info)
                title = info.get('title', 'Instagram Reel')
                return video_path, title
        except Exception as exc:
            last_exc = exc
            continue

    # Fallback to no cookies
    try:
        with ydl_lib.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
            title = info.get('title', 'Instagram Reel')
            return video_path, title
    except Exception as exc:
        raise last_exc or exc