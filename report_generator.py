"""
FedCrack-Net XAI: Master Structural Inspection Report Generator
Coordinates LLM report synthesis, HTML interactive preview generation,
and binary exports to PDF and DOCX.
"""

import base64
import io
from typing import Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image

from llm_service import LLMService
from pdf_export import PDFReportExporter
from docx_export import DOCXReportExporter
from report_templates import DOCUMENT_CSS


def _img_to_b64(arr: Optional[np.ndarray], max_dim: int = 450) -> Optional[str]:
    """Convert numpy RGB image array to base64 JPEG data URL for HTML embedding."""
    if arr is None or not isinstance(arr, np.ndarray) or arr.size == 0:
        return None
    try:
        pil_im = Image.fromarray(arr.astype(np.uint8))
        pil_im.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        pil_im.save(buf, format="JPEG", quality=85)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"
    except Exception:
        return None



class ReportGenerator:
    """Master orchestrator for AI-powered civil engineering inspection reports."""

    def __init__(self, api_key: Optional[str] = None):
        self.llm_service = LLMService(api_key=api_key)
        self.pdf_exporter = PDFReportExporter()
        self.docx_exporter = DOCXReportExporter()

    def generate_full_report_package(
        self,
        session_state_dict: Dict[str, Any],
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Build complete report package: structured report dict, HTML preview, PDF bytes, and DOCX bytes.
        """
        metadata = metadata or {}

        # 1. Extract empirical measurements from session state
        uploaded_img = session_state_dict.get("uploaded_image")
        img_name = session_state_dict.get("uploaded_image_name", "inspection_image.jpg")
        detections = session_state_dict.get("uploaded_detections") or []
        annotated_img = session_state_dict.get("uploaded_prediction")
        gradcam_img = session_state_dict.get("uploaded_gradcam")
        confidences = session_state_dict.get("uploaded_confidence") or []

        num_cracks = len(detections)

        if num_cracks > 0:
            top_det = detections[0]
            crack_type = top_det.get("crack_type", "Unknown Crack Type")
            severity = top_det.get("severity_level", "Moderate")
            conf = float(np.mean(confidences)) if confidences else top_det.get("confidence", 0.85)
            length = sum(d.get("crack_length_px", 0.0) for d in detections)
            avg_width = float(np.mean([d.get("avg_width_px", 2.0) for d in detections]))
            max_width = max([d.get("max_width_px", 3.0) for d in detections])
            angle = top_det.get("angle_deg", 0.0)
            curvature = top_det.get("curvature", 1.0)
            area_pct = sum(d.get("rel_area_pct", 0.0) for d in detections)
        else:
            crack_type = "None Detected"
            severity = "None"
            conf = 0.95
            length = 0.0
            avg_width = 0.0
            max_width = 0.0
            angle = 0.0
            curvature = 1.0
            area_pct = 0.0

        inspection_payload = {
            "image_name": img_name,
            "num_cracks": num_cracks,
            "crack_type": crack_type,
            "severity": severity,
            "confidence": conf,
            "length": length,
            "avg_width": avg_width,
            "max_width": max_width,
            "orientation": angle,
            "curvature": curvature,
            "area_coverage": area_pct,
            "detections_list": detections,
            "metadata": metadata,
        }

        # 2. Synthesize 10-section report via LLM Service (Offline / Online)
        report_data = self.llm_service.generate_report(inspection_payload)

        # 3. Generate HTML preview for Streamlit viewer
        html_preview = self.render_html_preview(
            report_data=report_data,
            original_img=uploaded_img,
            detection_img=annotated_img,
            gradcam_img=gradcam_img
        )

        # 4. Generate PDF Binary
        pdf_bytes = self.pdf_exporter.build_pdf(
            report_data=report_data,
            original_img=uploaded_img,
            detection_img=annotated_img,
            gradcam_img=gradcam_img
        )

        # 5. Generate DOCX Binary
        docx_bytes = self.docx_exporter.build_docx(
            report_data=report_data,
            original_img=uploaded_img,
            detection_img=annotated_img,
            gradcam_img=gradcam_img
        )

        return {
            "report_data": report_data,
            "html_preview": html_preview,
            "pdf_bytes": pdf_bytes,
            "docx_bytes": docx_bytes,
        }

    def render_html_preview(
        self,
        report_data: Dict[str, Any],
        original_img: Optional[np.ndarray] = None,
        detection_img: Optional[np.ndarray] = None,
        gradcam_img: Optional[np.ndarray] = None
    ) -> str:
        """Render clean, publication-styled white document HTML for in-app viewing."""
        info = report_data.get("inspection_info", {})
        summary = report_data.get("detection_summary", {})
        risk = report_data.get("risk_assessment", {})
        actions = report_data.get("recommended_actions", {})

        severity = summary.get("severity", "Moderate")
        badge_class = f"badge-{severity.lower()}" if severity.lower() in ["minor", "moderate", "severe", "critical"] else "badge-moderate"

        # Generate base64 images for visual plates
        plates_html = ""
        b64_orig = _img_to_b64(original_img)
        b64_det = _img_to_b64(detection_img)
        b64_cam = _img_to_b64(gradcam_img)

        if b64_orig or b64_det or b64_cam:
            plates_html = f"""
            <!-- Visual Evidence Plates -->
            <div class="section-title"><span>🖼️</span> Visual Inspection Evidence Plates</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin-bottom: 1.5rem; text-align: center;">
                <div style="background: #F8FAFC; padding: 10px; border-radius: 8px; border: 1px solid #E2E8F0;">
                    {f'<img src="{b64_orig}" style="width: 100%; border-radius: 6px; aspect-ratio: 1; object-fit: cover;" />' if b64_orig else '<div style="height: 180px; display:flex; align-items:center; justify-content:center; color:#94A3B8;">No Image</div>'}
                    <div style="font-size: 0.82rem; font-weight: 700; color: #1E3A8A; margin-top: 8px;">Plate I: Original Surface</div>
                </div>
                <div style="background: #F8FAFC; padding: 10px; border-radius: 8px; border: 1px solid #E2E8F0;">
                    {f'<img src="{b64_det}" style="width: 100%; border-radius: 6px; aspect-ratio: 1; object-fit: cover;" />' if b64_det else '<div style="height: 180px; display:flex; align-items:center; justify-content:center; color:#94A3B8;">No Image</div>'}
                    <div style="font-size: 0.82rem; font-weight: 700; color: #1E3A8A; margin-top: 8px;">Plate II: YOLOv8 Crack Detections</div>
                </div>
                <div style="background: #F8FAFC; padding: 10px; border-radius: 8px; border: 1px solid #E2E8F0;">
                    {f'<img src="{b64_cam}" style="width: 100%; border-radius: 6px; aspect-ratio: 1; object-fit: cover;" />' if b64_cam else '<div style="height: 180px; display:flex; align-items:center; justify-content:center; color:#94A3B8;">No Image</div>'}
                    <div style="font-size: 0.82rem; font-weight: 700; color: #1E3A8A; margin-top: 8px;">Plate III: Grad-CAM Explainable Saliency</div>
                </div>
            </div>
            """

        # Format causes
        causes_html = ""
        for line in report_data.get("possible_causes", "").split("\n"):
            line = line.strip()
            if line:
                if line.startswith("•") or line.startswith("*"):
                    causes_html += f"<li style='margin-bottom: 0.35rem;'>{line.lstrip('•* ')}</li>"
                else:
                    causes_html += f"<p style='margin-bottom: 0.5rem;'>{line}</p>"
        if "<li>" in causes_html:
            causes_html = causes_html.replace("<li", "<ul style='padding-left: 1.2rem;'><li", 1) + "</ul>"

        # Format risk details
        risk_html = ""
        for line in risk.get("details", "").split("\n"):
            line = line.strip()
            if line:
                risk_html += f"<p style='margin-bottom: 0.5rem;'>{line}</p>"

        # Format engineering notes
        notes_html = ""
        for line in report_data.get("engineering_notes", "").split("\n"):
            line = line.strip()
            if line:
                notes_html += f"<p style='margin-bottom: 0.4rem;'>{line}</p>"

        html = f"""
        {DOCUMENT_CSS}
        <div class="report-paper">
            <!-- 1. Header Banner -->
            <div class="report-header">
                <div>
                    <h1 class="report-title">AI Structural Inspection Report</h1>
                    <div class="report-subtitle">
                        <b>{report_data.get('project_name', 'FedCrack-Net XAI')}</b> — Federated Explainable Structural Crack Detection System
                    </div>
                </div>
                <div style="text-align: right;">
                    <span class="report-badge {badge_class}">Risk: {risk.get('risk_level', 'Moderate')}</span>
                    <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 0.3rem;">{report_data.get('project_version', 'v1.0')}</div>
                </div>
            </div>

            <!-- 2. Inspection Information -->
            <div class="section-title"><span>📋</span> 2. Inspection Information</div>
            <div class="meta-grid">
                <div class="meta-item">
                    <div class="meta-label">Inspection Date & Time</div>
                    <div class="meta-value">{info.get('date', 'N/A')} at {info.get('time', 'N/A')}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Asset / Structure Name</div>
                    <div class="meta-value">{info.get('building_name', 'N/A')}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Inspection Zone / Location</div>
                    <div class="meta-value">{info.get('location', 'N/A')}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Inspecting Entity</div>
                    <div class="meta-value">{info.get('inspector', 'AI Assistant')}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Asset Owner / Client</div>
                    <div class="meta-value">{info.get('client', 'N/A')}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Target Inspection Image</div>
                    <div class="meta-value">{info.get('image_name', 'N/A')}</div>
                </div>
            </div>

            {plates_html}

            <!-- 3. Empirical Detection Summary -->
            <div class="section-title"><span>📊</span> 3. Empirical Detection & Geometric Summary</div>
            <table class="report-table">
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Measured Metric</th>
                        <th>Parameter</th>
                        <th>Measured Metric</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><b>Identified Crack Count</b></td>
                        <td>{summary.get('num_cracks', 0)} defect(s)</td>
                        <td><b>Primary Crack Classification</b></td>
                        <td>{summary.get('crack_type', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td><b>Assessed Severity Tier</b></td>
                        <td><span class="report-badge {badge_class}">{summary.get('severity', 'N/A')}</span></td>
                        <td><b>Detection Confidence</b></td>
                        <td>{summary.get('confidence', 0)*100:.1f}%</td>
                    </tr>
                    <tr>
                        <td><b>Total Fissure Length</b></td>
                        <td>{summary.get('length_px', 0):.1f} px</td>
                        <td><b>Average Crack Aperture</b></td>
                        <td>{summary.get('avg_width_px', 0):.1f} px</td>
                    </tr>
                    <tr>
                        <td><b>Peak Maximum Aperture</b></td>
                        <td>{summary.get('max_width_px', 0):.1f} px</td>
                        <td><b>PCA Orientation Angle</b></td>
                        <td>{summary.get('orientation_deg', 0):.1f}°</td>
                    </tr>
                    <tr>
                        <td><b>Curvature Tortuosity Index</b></td>
                        <td>{summary.get('curvature', 1.0):.2f}</td>
                        <td><b>Surface Area Coverage</b></td>
                        <td>{summary.get('area_coverage_pct', 0):.2f}%</td>
                    </tr>
                </tbody>
            </table>

            <!-- 4. AI Structural Assessment -->
            <div class="section-title"><span>🔬</span> 4. AI Structural Assessment</div>
            <p style="text-align: justify; margin-bottom: 1.2rem;">{report_data.get('structural_assessment', '')}</p>

            <!-- 5. Possible Engineering Causes (Visual Assessment) -->
            <div class="section-title"><span>🔍</span> 5. Possible Engineering Causes (Visual Assessment)</div>
            <div style="background-color: #F8FAFC; padding: 1rem 1.2rem; border-radius: 8px; border: 1px solid #E2E8F0; margin-bottom: 1.2rem;">
                {causes_html}
            </div>

            <!-- 6. Risk Assessment -->
            <div class="section-title"><span>⚠️</span> 6. Risk Assessment & Durability Implications</div>
            <div style="margin-bottom: 1.2rem;">
                {risk_html}
            </div>

            <!-- 7. Recommended Actions -->
            <div class="section-title"><span>🛠️</span> 7. Recommended Engineering Actions</div>
            <div class="action-box">
                <div class="action-title">1. Immediate Phase (0 – 7 Days):</div>
                <div style="color: #334155; font-size: 0.9rem;">{actions.get('immediate', 'Establish photographic baseline')}</div>
            </div>
            <div class="action-box" style="border-left-color: #F59E0B; background-color: #FFFBEB;">
                <div class="action-title" style="color: #B45309;">2. Short-Term Phase (1 – 3 Months):</div>
                <div style="color: #334155; font-size: 0.9rem;">{actions.get('short_term', 'Install telltales and seal')}</div>
            </div>
            <div class="action-box" style="border-left-color: #10B981; background-color: #ECFDF5;">
                <div class="action-title" style="color: #065F46;">3. Long-Term Phase (Structural Restoration):</div>
                <div style="color: #334155; font-size: 0.9rem;">{actions.get('long_term', 'Low pressure resin injection')}</div>
            </div>

            <!-- 8. Explainable AI (Grad-CAM) -->
            <div class="section-title"><span>💡</span> 8. Explainable AI (Grad-CAM) Verification</div>
            <p style="text-align: justify; margin-bottom: 1.2rem;">{report_data.get('xai_explanation', '')}</p>

            <!-- 9. Engineering Notes -->
            <div class="section-title"><span>📌</span> 9. Engineering Notes</div>
            <div style="color: #475569; font-size: 0.9rem; line-height: 1.6; margin-bottom: 1.2rem;">
                {notes_html}
            </div>

            <!-- 10. Disclaimer -->
            <div class="disclaimer-box">
                <b>10. Regulatory Disclaimer:</b> {report_data.get('disclaimer', '')}
            </div>

            <!-- Footer -->
            <div class="report-footer">
                <div>FedCrack-Net XAI | Generated by AI Structural Assistant</div>
                <div>Confidential Engineering Document</div>
            </div>
        </div>
        """
        return html
