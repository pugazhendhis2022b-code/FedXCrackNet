"""
FedCrack-Net XAI: PDF Inspection Report Exporter
Generates publication-quality civil engineering inspection reports in PDF format
using ReportLab, with structured tables, inspection visual plates, and running footers.
"""

import io
import os
from typing import Dict, Any, Optional, List
import numpy as np
from PIL import Image as PILImage

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for dynamic total page numbering."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "FedCrack-Net XAI — AI Structural Inspection Report")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Running Footer
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, "Generated using FedCrack-Net XAI | Confidential Civil Engineering Assessment")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 32, page_str)
        self.restoreState()


class PDFReportExporter:
    """Compiles structured inspection reports into formatted PDF documents."""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        self.primary_blue = colors.HexColor("#1E3A8A")
        self.secondary_blue = colors.HexColor("#2563EB")
        self.text_dark = colors.HexColor("#0F172A")
        self.text_muted = colors.HexColor("#475569")
        self.bg_light = colors.HexColor("#F8FAFC")

        self.title_style = ParagraphStyle(
            "DocTitle",
            parent=self.styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=self.primary_blue,
            spaceAfter=4
        )
        self.subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=self.text_muted,
            spaceAfter=12
        )
        self.h2_style = ParagraphStyle(
            "DocH2",
            parent=self.styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=self.primary_blue,
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True
        )
        self.body_style = ParagraphStyle(
            "DocBody",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13.5,
            textColor=self.text_dark,
            spaceAfter=6
        )
        self.callout_style = ParagraphStyle(
            "DocCallout",
            parent=self.styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=12.5,
            textColor=self.text_muted,
            spaceAfter=6
        )
        self.table_cell = ParagraphStyle(
            "TableCell",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11.5,
            textColor=self.text_dark
        )
        self.table_header = ParagraphStyle(
            "TableHeader",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11.5,
            textColor=colors.white
        )

    def _convert_numpy_to_temp_img(self, img_arr: np.ndarray, max_width: float = 2.2) -> Optional[RLImage]:
        """Convert numpy RGB array to in-memory ReportLab Image flowable."""
        try:
            pil_img = PILImage.fromarray(img_arr)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            orig_w, orig_h = pil_img.size
            aspect = orig_h / max(1, orig_w)
            w = max_width * inch
            h = w * aspect
            return RLImage(buf, width=w, height=h)
        except Exception as e:
            print(f"[PDFExport] Image conversion warning: {e}")
            return None

    def build_pdf(
        self,
        report_data: Dict[str, Any],
        original_img: Optional[np.ndarray] = None,
        detection_img: Optional[np.ndarray] = None,
        gradcam_img: Optional[np.ndarray] = None
    ) -> bytes:
        """Generate PDF document binary buffer from report data and visual images."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=54,
            rightMargin=54,
            topMargin=54,
            bottomMargin=54
        )

        story = []

        # 1. Header & Title Banner
        story.append(Paragraph("AI Structural Inspection Report", self.title_style))
        story.append(Paragraph("<b>FedCrack-Net XAI</b> — Federated Explainable Structural Crack Detection System", self.subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=self.primary_blue, spaceBefore=0, spaceAfter=12))

        # 2. Inspection Information Table
        info = report_data.get("inspection_info", {})
        info_data = [
            [
                Paragraph("<b>Date:</b> " + str(info.get("date", "N/A")), self.table_cell),
                Paragraph("<b>Time:</b> " + str(info.get("time", "N/A")), self.table_cell),
                Paragraph("<b>Inspector:</b> " + str(info.get("inspector", "AI Assistant")), self.table_cell),
            ],
            [
                Paragraph("<b>Building / Structure:</b> " + str(info.get("building_name", "Concrete Asset")), self.table_cell),
                Paragraph("<b>Location:</b> " + str(info.get("location", "Inspection Zone")), self.table_cell),
                Paragraph("<b>Client / Owner:</b> " + str(info.get("client", "Asset Owner")), self.table_cell),
            ],
            [
                Paragraph("<b>Image File:</b> " + str(info.get("image_name", "N/A")), self.table_cell),
                Paragraph("<b>Framework Version:</b> " + str(report_data.get("project_version", "v1.0")), self.table_cell),
                Paragraph("<b>Assessment Type:</b> Preliminary Visual XAI", self.table_cell),
            ],
        ]
        info_table = Table(info_data, colWidths=[168, 168, 168])
        info_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), self.bg_light),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 10))

        # Visual Plates (Images) side-by-side if available
        visual_flowables = []
        if original_img is not None:
            rl_orig = self._convert_numpy_to_temp_img(original_img, max_width=2.2)
            if rl_orig: visual_flowables.append(("1. Input Photo", rl_orig))
        if detection_img is not None:
            rl_det = self._convert_numpy_to_temp_img(detection_img, max_width=2.2)
            if rl_det: visual_flowables.append(("2. YOLOv8 Detections", rl_det))
        if gradcam_img is not None:
            rl_cam = self._convert_numpy_to_temp_img(gradcam_img, max_width=2.2)
            if rl_cam: visual_flowables.append(("3. Grad-CAM Saliency", rl_cam))

        if visual_flowables:
            story.append(Paragraph("Inspection Visual Plates", self.h2_style))
            img_cells = []
            caption_cells = []
            for cap, flowable in visual_flowables:
                img_cells.append(flowable)
                caption_cells.append(Paragraph(f"<b>{cap}</b>", self.callout_style))
            img_table = Table([img_cells, caption_cells], colWidths=[168] * len(visual_flowables))
            img_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
            ]))
            story.append(img_table)
            story.append(Spacer(1, 10))

        # 3. Detection Summary Table
        summary = report_data.get("detection_summary", {})
        story.append(Paragraph("3. Empirical Detection & Geometric Summary", self.h2_style))

        sum_data = [
            [
                Paragraph("<b>Parameter</b>", self.table_header),
                Paragraph("<b>Value</b>", self.table_header),
                Paragraph("<b>Parameter</b>", self.table_header),
                Paragraph("<b>Value</b>", self.table_header)
            ],
            [
                Paragraph("Cracks Identified", self.table_cell),
                Paragraph(str(summary.get("num_cracks", 0)), self.table_cell),
                Paragraph("Primary Crack Type", self.table_cell),
                Paragraph(str(summary.get("crack_type", "N/A")), self.table_cell),
            ],
            [
                Paragraph("Severity Classification", self.table_cell),
                Paragraph(str(summary.get("severity", "N/A")), self.table_cell),
                Paragraph("Detection Confidence", self.table_cell),
                Paragraph(f"{summary.get('confidence', 0)*100:.1f}%", self.table_cell),
            ],
            [
                Paragraph("Fissure Length", self.table_cell),
                Paragraph(f"{summary.get('length_px', 0):.1f} px", self.table_cell),
                Paragraph("Average Aperture", self.table_cell),
                Paragraph(f"{summary.get('avg_width_px', 0):.1f} px", self.table_cell),
            ],
            [
                Paragraph("Maximum Aperture", self.table_cell),
                Paragraph(f"{summary.get('max_width_px', 0):.1f} px", self.table_cell),
                Paragraph("Orientation Angle", self.table_cell),
                Paragraph(f"{summary.get('orientation_deg', 0):.1f}°", self.table_cell),
            ],
            [
                Paragraph("Curvature Tortuosity", self.table_cell),
                Paragraph(f"{summary.get('curvature', 1.0):.2f}", self.table_cell),
                Paragraph("Area Coverage", self.table_cell),
                Paragraph(f"{summary.get('area_coverage_pct', 0):.2f}%", self.table_cell),
            ]
        ]
        summary_table = Table(sum_data, colWidths=[130, 122, 130, 122])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), self.primary_blue),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, self.bg_light]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 10))

        # 4. AI Structural Assessment
        story.append(Paragraph("4. AI Structural Assessment", self.h2_style))
        story.append(Paragraph(report_data.get("structural_assessment", ""), self.body_style))

        # 5. Possible Engineering Causes (Visual Assessment)
        story.append(Paragraph("5. Possible Engineering Causes (Visual Assessment)", self.h2_style))
        for line in report_data.get("possible_causes", "").split("\n"):
            line = line.strip()
            if line:
                if line.startswith("•") or line.startswith("*"):
                    story.append(Paragraph(f"&nbsp;&nbsp;{line}", self.body_style))
                else:
                    story.append(Paragraph(line, self.body_style))

        # 6. Risk Assessment
        risk_info = report_data.get("risk_assessment", {})
        story.append(Paragraph("6. Risk Assessment & Durability Implications", self.h2_style))
        for line in risk_info.get("details", "").split("\n"):
            line = line.strip()
            if line:
                story.append(Paragraph(line, self.body_style))

        # 7. Recommended Actions
        actions = report_data.get("recommended_actions", {})
        story.append(Paragraph("7. Recommended Engineering Actions", self.h2_style))
        act_data = [
            [Paragraph("<b>Phase</b>", self.table_header), Paragraph("<b>Recommended Engineering Measure</b>", self.table_header)],
            [Paragraph("<b>Immediate</b>", self.table_cell), Paragraph(actions.get("immediate", "Observe"), self.table_cell)],
            [Paragraph("<b>Short-Term</b>", self.table_cell), Paragraph(actions.get("short_term", "Seal"), self.table_cell)],
            [Paragraph("<b>Long-Term</b>", self.table_cell), Paragraph(actions.get("long_term", "Restore"), self.table_cell)],
        ]
        act_table = Table(act_data, colWidths=[100, 404])
        act_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), self.primary_blue),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, self.bg_light]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(act_table)
        story.append(Spacer(1, 8))

        # 8. Explainable AI (Grad-CAM)
        story.append(Paragraph("8. Explainable AI (Grad-CAM) Verification", self.h2_style))
        story.append(Paragraph(report_data.get("xai_explanation", ""), self.body_style))

        # 9. Engineering Notes
        story.append(Paragraph("9. Engineering Notes", self.h2_style))
        for line in report_data.get("engineering_notes", "").split("\n"):
            line = line.strip()
            if line:
                story.append(Paragraph(line, self.body_style))

        # 10. Regulatory Disclaimer
        story.append(Paragraph("10. Regulatory Disclaimer", self.h2_style))
        disc_text = report_data.get("disclaimer", "")
        disc_table = Table([[Paragraph(f"<i>{disc_text}</i>", self.callout_style)]], colWidths=[504])
        disc_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(disc_table)

        doc.build(story, canvasmaker=NumberedCanvas)
        return buffer.getvalue()
