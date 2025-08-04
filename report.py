from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image, PageBreak
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from datetime import datetime
import os
import matplotlib.pyplot as plt
import tempfile
from collections import Counter
import numpy as np

def safe_float(val, decimals=2):
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return f"{0:.{decimals}f}"

def format_summary_key(key):
    """Format summary keys for better display"""
    key_mapping = {
        'total_records': 'Total Records',
        'avg_speed': 'Average Speed (km/h)',
        'top_speed': 'Top Speed (km/h)',
        'lowest_speed': 'Lowest Speed (km/h)',
        'most_detected_object': 'Most Detected Object',
        'approaching_count': 'Approaching Count',
        'stationary_count': 'Stationary Count',
        'departing_count': 'Departing Count',
        'last_detection': 'Last Detection'
    }
    return key_mapping.get(key, key.replace('_', ' ').title())

def draw_chart_image(title, labels, data):
    try:
        fig, ax = plt.subplots(figsize=(6.8, 2.4))
        ax.bar(labels, data, color='steelblue')
        ax.set_title(title)
        ax.set_ylabel('Count')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
        ax.grid(True, linestyle='--', alpha=0.5)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.tight_layout()
        plt.savefig(temp_file.name, dpi=100)
        plt.close(fig)
        return temp_file.name
    except Exception as e:
        print(f"Chart error: {e}")
        return None

def add_header_footer(canvas, doc, logo_path):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    page_num = doc.page
    canvas.drawCentredString(420, 20, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}   |   Page {page_num}")
    if logo_path and os.path.exists(logo_path):
        canvas.drawImage(logo_path, x=30, y=10, width=50, height=30, preserveAspectRatio=True)
    canvas.restoreState()

def generate_pdf_report(
    filepath,
    title="Radar Based Speed Detection Report",
    logo_path="/home/pi/iwr6843isk/static/essi_logo.jpeg",
    summary=None,
    data=None,
    filters=None,
    charts=None
):
    styles = getSampleStyleSheet()
    centered_title = ParagraphStyle('CenteredTitle', parent=styles['Title'], alignment=1, fontSize=18, spaceAfter=6)

    content = []

    header_style = ParagraphStyle('header', fontSize=14, alignment=TA_CENTER, fontName='Helvetica-Bold', spaceAfter=4)
    subheader_style = ParagraphStyle('subheader', fontSize=11, alignment=TA_CENTER, fontName='Helvetica', spaceAfter=6)

    if logo_path and os.path.exists(logo_path):
        logo = Image(logo_path, width=2.0*inch, height=0.7*inch)
        logo.hAlign = 'CENTER'
        content.append(logo)

    content.append(Spacer(1, 6))
    content.append(Paragraph(f"<b>{title}</b>", centered_title))
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    content.append(Paragraph(f"Generated on {now_str}", ParagraphStyle("datetime", alignment=1, fontSize=10)))
    content.append(Spacer(1, 12))

    if summary:
        styleN = styles['Normal']
        summary_copy = summary.copy()
        speed_limits_raw = summary_copy.pop("speed_limits", {})
        speed_limits = {k.upper(): v for k, v in speed_limits_raw.items()}
        if not isinstance(speed_limits, dict) or not speed_limits:
            speed_limits = {
                'human': 8,
                'car': 60,
                'truck': 50,
                'bus': 50,
                'bike': 60,
                'bicycle': 10,
                'unknown': 50,
                'default': 50
            }

        # Ordered summary fields as per required format
        ordered_fields = [
            ('total_records', 'Total Records'),
            ('avg_speed', 'Average Speed (km/h)'),
            ('top_speed', 'Top Speed (km/h)'),
            ('lowest_speed', 'Lowest Speed (km/h)'),
            ('most_detected_object', 'Most Detected Object'),
            ('approaching_count', 'Approaching Count'),
            ('stationary_count', 'Stationary Count'),
            ('departing_count', 'Departing Count'),
            ('last_detection', 'Last Detection')
        ]

        summary_data = []
        for key, label in ordered_fields:
            val = summary_copy.get(key, "N/A")
            if isinstance(val, (int, float)):
                val = safe_float(val)
            elif not val:
                val = "N/A"
            summary_data.append([
                Paragraph(f"<b>{label}</b>", styleN),
                Paragraph(str(val), styleN)
            ])

        summary_table = Table(summary_data, colWidths=[2.8 * inch, 2.0 * inch])
        summary_table.setStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ])

        # Build object type → speed limit table in required order
        object_types = ['HUMAN', 'CAR', 'TRUCK', 'BUS', 'BIKE', 'BICYCLE', 'UNKNOWN', 'DEFAULT']
        speed_data = [[Paragraph("<b>Object Type</b>", styleN), Paragraph("<b>Speed Limit (km/h)</b>", styleN)]]
        for obj in object_types:
            limit = speed_limits.get(obj.upper(), "N/A")
            limit = safe_float(limit) if isinstance(limit, (int, float)) else "N/A"
            speed_data.append([
                Paragraph(obj.title(), styleN),
                Paragraph(limit, styleN)
            ])

        speed_table = Table(speed_data, colWidths=[2.4 * inch, 2.4 * inch])
        speed_table.setStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ])

        content.append(Paragraph("<b>Summary Statistics</b>", styles['Heading3']))
        content.append(Spacer(1, 6))
        content.append(summary_table)
        content.append(Spacer(1, 12))

        content.append(Spacer(1, 6))
        content.append(speed_table)
        content.append(Spacer(1, 12))


    if charts:
        valid_charts = []
        for chart_title, chart_data in charts.items():
            labels = chart_data.get("labels", [])
            data_vals = chart_data.get("data", [])
            if labels and data_vals and any(val > 0 for val in data_vals) and len(labels) == len(data_vals):
                valid_charts.append((chart_title, labels, data_vals))

        if valid_charts:
            content.append(PageBreak())
            content.append(Paragraph("<b>Analytics Charts</b>", styles['Heading3']))
            content.append(Spacer(1, 6))
            for chart_title, labels, data_vals in valid_charts:
                chart_img = draw_chart_image(chart_title.replace("_", " ").title(), labels, data_vals)
                if chart_img and os.path.exists(chart_img):
                    content.append(Image(chart_img, width=6.8 * inch, height=2.2 * inch))
                    content.append(Spacer(1, 6))

    if data:
        content.append(Paragraph("<b>Detection Data</b>", styles['Heading3']))
        content.append(Spacer(1, 6))
        
        headers = [
            "Snapshot", "Datetime", "Sensor", "Object ID", "Type", "Confidence", "Speed (km/h)",
            "Velocity", "Distance", "Radar Dist", "Visual Dist", "Direction",
            "Signal (dB)", "Doppler Freq (Hz)", "Reviewed", "Flagged"
        ]
        table_data = [headers]

        for row in data:
            thumb_path = row.get("snapshot_path")
            try:
                if thumb_path and os.path.exists(thumb_path):
                    img = Image(thumb_path, width=0.6 * inch, height=0.4 * inch)
                else:
                    img = "N/A"
            except Exception as e:
                print(f"[SNAPSHOT ERROR] {thumb_path} → {e}")
                img = "N/A"

            table_data.append([
                img,
                row.get("datetime", "N/A"),
                row.get("sensor", "N/A"),
                row.get("object_id", "N/A"),
                row.get("type", "UNKNOWN"),
                safe_float(row.get('confidence')),
                safe_float(row.get('speed_kmh')),
                safe_float(row.get('velocity')),
                safe_float(row.get('distance')),
                safe_float(row.get('radar_distance')),
                safe_float(row.get('visual_distance')),
                row.get("direction", "N/A"),
                safe_float(row.get('signal_level'), 1),
                safe_float(row.get('doppler_frequency')),
                "Yes" if row.get("reviewed") else "No",
                "Yes" if row.get("flagged") else "No",
            ])

        table = Table(table_data, repeatRows=1, colWidths=[0.8*inch] + [None]*15)
        table.setStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
        content.append(table)

    doc = BaseDocTemplate(filepath, pagesize=landscape(A4), rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')
    template = PageTemplate(id='withHeaderFooter', frames=frame, onPage=lambda c, d: add_header_footer(c, d, logo_path))
    doc.addPageTemplates([template])
    doc.build(content)
    return filepath
