from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate,
    Paragraph, Spacer, Table, Image, PageBreak, KeepTogether
)
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tempfile

# ---------------- helpers ----------------

def safe_float(val, decimals=2):
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return f"{0:.{decimals}f}"

def _to_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return float(default)

def _speed_limits_from(summary):
    raw = (summary or {}).get("speed_limits", {}) or {}
    try:
        lim = {str(k).upper(): float(v) for k, v in raw.items()}
    except Exception:
        lim = {}
    if not lim:
        lim = {"HUMAN": 8, "CAR": 60, "TRUCK": 50, "BUS": 50, "BIKE": 60, "BICYCLE": 10, "UNKNOWN": 50}
    if "DEFAULT" not in lim:
        lim["DEFAULT"] = lim.get("UNKNOWN", 50)
    return lim

def _is_speeding(row, limits):
    t = str(row.get("type", "UNKNOWN")).upper()
    spd = _to_float(row.get("speed_kmh"), 0)
    return spd > limits.get(t, limits.get("DEFAULT", 50))

def _tmp_png():
    return tempfile.NamedTemporaryFile(delete=False, suffix=".png").name

def draw_chart_image(title, labels, data):
    try:
        if not labels or not data or len(labels) != len(data):
            return None
        if not any((_to_float(v) > 0) for v in data):
            return None
        fig, ax = plt.subplots(figsize=(6.8, 2.4))
        ax.bar(range(len(labels)), data)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.4)
        f = _tmp_png()
        plt.tight_layout()
        plt.savefig(f, dpi=110)
        plt.close(fig)
        return f
    except Exception as e:
        print(f"[Chart error] {e}")
        return None

class NumberedCanvas(canvas.Canvas):
    """Footer: 'Generated on ... | Page X of Y'"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_footer(total)
            super().showPage()
        super().save()

    def _draw_footer(self, page_count):
        self.setFont("Helvetica", 8)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.drawCentredString(420, 18, f"Generated on {ts}  |  Page {self._pageNumber} of {page_count}")

def add_header_footer(c, doc, logo_path):
    c.saveState()
    if logo_path and os.path.exists(logo_path):
        try:
            c.drawImage(logo_path, x=doc.leftMargin, y=doc.height + doc.bottomMargin + 6,
                        width=60, height=22, preserveAspectRatio=True, mask='auto')
        except Exception:
            pass
    c.restoreState()

# ---------------- ui blocks ----------------

def _kpi_cards(summary):
    styles = getSampleStyleSheet()
    n = styles["Normal"]; n.fontSize = 9
    center = ParagraphStyle('center', parent=n, alignment=TA_CENTER)
    big = ParagraphStyle('big', parent=center, fontSize=14)
    muted = ParagraphStyle('muted', parent=center, fontSize=8, textColor=colors.HexColor('#6B7280'))

    def tile(label, value, sub=""):
        title_par = Paragraph(f"<b>{xml_escape(str(label))}</b>", center)
        big_par = Paragraph(f"<b>{xml_escape(str(value))}</b>", big)
        sub_par = Paragraph(xml_escape(str(sub)), muted)
        card = Table([[title_par],[big_par],[sub_par]], colWidths=[2.2*inch],
                     style=[("BACKGROUND",(0,0),(-1,-1), colors.white),
                            ("BOX",(0,0),(-1,-1),0.6, colors.HexColor("#D0D7DE")),
                            ("VALIGN",(0,0),(-1,-1),"MIDDLE")])
        return card

    total = summary.get("total_records", 0)
    avg = safe_float(summary.get("avg_speed", 0.0))
    top = safe_float(summary.get("top_speed", 0.0))
    low = safe_float(summary.get("lowest_speed", 0.0))
    auto = summary.get("auto_snapshots", 0)
    manual = summary.get("manual_snapshots", 0)
    approach = summary.get("approaching_count", 0)
    depart = summary.get("departing_count", 0)
    stationary = summary.get("stationary_count", 0)

    tiles = [
        tile("Total Records", total),
        tile("Average Speed (km/h)", avg),
        tile("Top Speed (km/h)", top),
        tile("Lowest Speed (km/h)", low),
        tile("Auto / Manual", f"{auto} / {manual}"),
        tile("Approach/Depart/Stat", f"{approach}/{depart}/{stationary}"),
    ]

    rows = [tiles[i:i+3] for i in range(0, len(tiles), 3)]
    for r in rows:
        while len(r) < 3:
            r.append(Spacer(1,1))  # placeholder must be a Flowable
    return Table(rows, colWidths=[2.4*inch]*3, hAlign="LEFT",
                 style=[("BOTTOMPADDING",(0,0),(-1,-1),6)])

def _filters_table(filters):
    if not filters:
        return Paragraph("<i>No filters applied (full dataset).</i>", getSampleStyleSheet()["Italic"])
    n = getSampleStyleSheet()["Normal"]
    rows = [[Paragraph("<b>Filter</b>", n), Paragraph("<b>Value</b>", n)]]
    for k, v in filters.items():
        rows.append([Paragraph(xml_escape(str(k)).replace("_"," ").title(), n),
                     Paragraph(xml_escape(str(v)), n)])
    t = Table(rows, colWidths=[2.6*inch, 5.6*inch])
    t.setStyle([
        ("GRID",(0,0),(-1,-1),0.4,colors.lightgrey),
        ("BACKGROUND",(0,0),(-1,-1),colors.whitesmoke),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("FONTSIZE",(0,0),(-1,-1),9),
        ("ALIGN",(0,0),(-1,-1),"LEFT"),
        ("LEFTPADDING",(0,0),(-1,-1),6),
        ("RIGHTPADDING",(0,0),(-1,-1),6),
    ])
    return t

def _speed_limits_table(limits):
    n = getSampleStyleSheet()["Normal"]
    rows = [[Paragraph("<b>Object Type</b>", n), Paragraph("<b>Speed Limit (km/h)</b>", n)]]
    order = ["HUMAN","CAR","TRUCK","BUS","BIKE","BICYCLE","UNKNOWN","DEFAULT"]
    for k in order:
        val = limits.get(k, limits.get("DEFAULT", 50))
        rows.append([Paragraph(xml_escape(k.title()), n), Paragraph(xml_escape(safe_float(val, 0)), n)])
    t = Table(rows, colWidths=[3.0*inch, 2.2*inch])
    t.setStyle([
        ("GRID",(0,0),(-1,-1),0.4,colors.lightgrey),
        ("BACKGROUND",(0,0),(-1,0),colors.lightblue),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("FONTSIZE",(0,0),(-1,-1),9),
    ])
    return t

def _snapshot_gallery(data, max_items=12, cols=3):
    items = []
    for r in (data or []):
        p = r.get("snapshot_path")
        if p and os.path.exists(p):
            cap = f"{str(r.get('type','UNK')).upper()} | {safe_float(r.get('speed_kmh'),1)} km/h | {safe_float(r.get('distance'),1)} m | {r.get('datetime','N/A')}"
            items.append((p, cap))
        if len(items) >= max_items:
            break
    if not items:
        return None

    cells, row = [], []
    col_w = 3.2*inch
    for i, (path, caption) in enumerate(items, 1):
        try:
            img = Image(path, width=col_w*0.95, height=col_w*0.6, kind='proportional')
        except Exception:
            img = Paragraph("N/A", getSampleStyleSheet()["BodyText"])
        cap = Paragraph(xml_escape(caption), ParagraphStyle('cap', parent=getSampleStyleSheet()["BodyText"], fontSize=7))
        cell = KeepTogether([img, Spacer(1,2), cap])
        row.append(cell)
        if i % cols == 0:
            cells.append(row); row = []
    if row:
        while len(row) < cols:
            row.append(Spacer(1,1))  # placeholder Flowable
        cells.append(row)

    t = Table(cells, colWidths=[col_w]*cols)
    t.setStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),
                ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                ("BOTTOMPADDING",(0,0),(-1,-1),6)])
    return t

# --------------- main ----------------

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
    centered_title = ParagraphStyle('CenteredTitle', parent=styles['Title'], alignment=TA_CENTER, fontSize=18, spaceAfter=6)

    content = []

    # Cover
    if logo_path and os.path.exists(logo_path):
        logo = Image(logo_path, width=2.0*inch, height=0.7*inch)
        logo.hAlign = 'CENTER'
        content.append(logo)
    content.append(Spacer(1, 6))
    content.append(Paragraph(xml_escape(f"{title}"), centered_title))
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    content.append(Paragraph(xml_escape(f"Generated on {now_str}"), ParagraphStyle("datetime", alignment=TA_CENTER, fontSize=10)))
    content.append(Spacer(1, 12))

    # KPI + Speed limits + Filters
    limits = _speed_limits_from(summary or {})
    if summary:
        content.append(_kpi_cards(summary))
        content.append(Spacer(1, 10))
        content.append(Paragraph("<b>Configured Speed Limits</b>", styles['Heading3']))
        content.append(Spacer(1, 4))
        content.append(_speed_limits_table(limits))
        content.append(Spacer(1, 10))

    if filters:
        content.append(Paragraph("<b>Filters</b>", styles['Heading3']))
        content.append(Spacer(1, 4))
        content.append(_filters_table(filters))
        content.append(Spacer(1, 10))

    # Charts
    if charts and isinstance(charts, dict):
        valid = []
        for chart_title, chart_data in charts.items():
            labels = chart_data.get("labels", [])
            vals = chart_data.get("data", [])
            img = draw_chart_image(chart_title.replace("_"," ").title(), labels, vals)
            if img and os.path.exists(img):
                valid.append(img)
        if valid:
            content.append(PageBreak())
            content.append(Paragraph("<b>Analytics Charts</b>", styles['Heading3']))
            content.append(Spacer(1, 6))
            for img in valid:
                content.append(Image(img, width=6.8*inch, height=2.2*inch))
                content.append(Spacer(1, 6))

    # Snapshot gallery
    gal = _snapshot_gallery(data or [])
    if gal:
        content.append(PageBreak())
        content.append(Paragraph("<b>Recent Snapshots</b>", styles['Heading3']))
        content.append(Spacer(1, 6))
        content.append(gal)
        content.append(Spacer(1, 6))

    # Data table (base + dynamic extra columns)
    if data:
        content.append(PageBreak())
        content.append(Paragraph("<b>Detection Data</b>", styles['Heading3']))
        content.append(Spacer(1, 6))

        # Base headers
        headers = [
            "Snapshot", "Datetime", "Sensor", "Object ID", "Type", "Confidence",
            "Speed (km/h)", "Velocity (m/s)", "Distance (m)",
            "Direction", "Motion State", "Signal (dB)", "Doppler (Hz)",
            "Reviewed", "Flagged"
        ]

        # Dynamic extras if present
        has_azimuth   = any(("azimuth"   in r and r.get("azimuth")   is not None) for r in data)
        has_elevation = any(("elevation" in r and r.get("elevation") is not None) for r in data)
        has_snr       = any(("snr"       in r and r.get("snr")       is not None) for r in data)
        has_gain      = any(("gain"      in r and r.get("gain")      is not None) for r in data)

        if has_azimuth:   headers.append("Azimuth (°)")
        if has_elevation: headers.append("Elevation (°)")
        if has_snr:       headers.append("SNR (dB)")
        if has_gain:      headers.append("Gain (dB)")

        table_data = [headers]
        speeding_rows = []

        for row in data:
            # Snapshot cell
            thumb_path = row.get("snapshot_path")
            try:
                if thumb_path and os.path.exists(thumb_path):
                    img = Image(thumb_path, width=0.7 * inch, height=0.5 * inch)
                else:
                    img = Paragraph("N/A", getSampleStyleSheet()["BodyText"])
            except Exception as e:
                print(f"[SNAPSHOT ERROR] {thumb_path} → {e}")
                img = Paragraph("N/A", getSampleStyleSheet()["BodyText"])

            if _is_speeding(row, limits):
                speeding_rows.append(len(table_data))

            base_values = [
                img,
                xml_escape(str(row.get("datetime", "N/A"))),
                xml_escape(str(row.get("sensor", "N/A"))),
                xml_escape(str(row.get("object_id", "N/A"))),
                xml_escape(str(row.get("type", "UNKNOWN"))),
                safe_float(row.get('confidence')),
                safe_float(row.get('speed_kmh')),
                safe_float(row.get('velocity')),
                safe_float(row.get('distance')),
                xml_escape(str(row.get("direction", "N/A"))),
                xml_escape(str(row.get("motion_state", "N/A"))),
                safe_float(row.get('signal_level'), 1),
                safe_float(row.get('doppler_frequency')),
                "Yes" if row.get("reviewed") else "No",
                "Yes" if row.get("flagged") else "No",
            ]

            if has_azimuth:   base_values.append(safe_float(row.get("azimuth")))
            if has_elevation: base_values.append(safe_float(row.get("elevation")))
            if has_snr:       base_values.append(safe_float(row.get("snr"), 1))
            if has_gain:      base_values.append(safe_float(row.get("gain"), 1))

            table_data.append(base_values)

        table = Table(table_data, repeatRows=1, colWidths=[0.85*inch] + [None]*(len(headers)-1))
        style = [
            ("BACKGROUND", (0,0), (-1,0), colors.darkblue),
            ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("FONTSIZE", (0,0), (-1,-1), 7),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ]
        # Zebra striping + highlight speeding rows and speed column (index 6)
        for i in range(1, len(table_data)):
            if i % 2 == 1:
                style.append(("BACKGROUND", (0,i), (-1,i), colors.HexColor("#F8FAFF")))
        for r in speeding_rows:
            style.append(("BACKGROUND", (0,r), (-1,r), colors.HexColor("#FFF0F0")))
            style.append(("TEXTCOLOR", (6,r), (6,r), colors.HexColor("#B00020")))
        table.setStyle(style)
        content.append(table)

    # Build
    doc = BaseDocTemplate(
        filepath,
        pagesize=landscape(A4),
        rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')
    template = PageTemplate(id='withHeaderFooter', frames=frame, onPage=lambda c, d: add_header_footer(c, d, logo_path))
    doc.addPageTemplates([template])
    doc.build(content, canvasmaker=NumberedCanvas)
    return filepath
