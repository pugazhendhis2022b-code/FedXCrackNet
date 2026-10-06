import sys
from pathlib import Path
BASE_DIR = Path(r"d:/final crack detection")
sys.path.insert(0, str(BASE_DIR))

import numpy as np
from report_generator import ReportGenerator

print("Testing ReportGenerator...")
generator = ReportGenerator()

dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
mock_session_state = {
    "uploaded_image": dummy_img,
    "uploaded_image_name": "test_wall_pier.jpg",
    "uploaded_prediction": dummy_img,
    "uploaded_gradcam": dummy_img,
    "uploaded_detections": [
        {
            "id": 1,
            "crack_type": "Horizontal Crack",
            "severity_level": "Moderate",
            "confidence": 0.912,
            "crack_length_px": 540.5,
            "avg_width_px": 4.2,
            "max_width_px": 6.8,
            "angle_deg": 14.5,
            "curvature": 1.45,
            "rel_area_pct": 0.85,
        }
    ],
    "uploaded_confidence": [0.912]
}

metadata = {
    "building_name": "Metro Viaduct Pier #14",
    "location": "Soffit Cantilever, Bay 2",
    "inspector": "Er. P. Sharma (Structural Inspector)",
    "client": "State Rapid Transit Authority"
}

pkg = generator.generate_full_report_package(mock_session_state, metadata=metadata)

report = pkg["report_data"]
print("Title:", report["title"])
print("Project:", report["project_name"])
print("Inspector:", report["inspection_info"]["inspector"])

assert "structural_assessment" in report
assert "possible_causes" in report
assert "Based on the visual characteristics of the detected crack, possible engineering causes may include" in report["possible_causes"]
assert "risk_assessment" in report
assert report["risk_assessment"]["risk_level"] in ["Low", "Moderate", "High", "Critical"]
assert "recommended_actions" in report
assert "xai_explanation" in report
assert "engineering_notes" in report
assert "disclaimer" in report

print("Report Data Sections Verified Successfully!")
print("HTML Preview Length:", len(pkg["html_preview"]))
print("PDF Bytes Length:", len(pkg["pdf_bytes"]))
print("DOCX Bytes Length:", len(pkg["docx_bytes"]))

assert len(pkg["pdf_bytes"]) > 1000
assert len(pkg["docx_bytes"]) > 1000

print("ALL REPORT GENERATION TESTS PASSED!")
