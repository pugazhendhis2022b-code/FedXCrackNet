import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
from PIL import Image
import cv2

from streamlit_app import load_framework_components, process_and_diagnose
from report_generator import ReportGenerator

components = load_framework_components()
loader = components["loader"]
test_imgs = loader.get_image_files("test")
assert len(test_imgs) > 0, "No test images found"

img_path = test_imgs[0]
img_pil = Image.open(img_path).convert("RGB")
img_np = np.array(img_pil)

print(f"Loaded test image: {img_path.name}")
results = process_and_diagnose(
    img_rgb=img_np,
    components=components,
    conf_thresh=0.15,
    iou_thresh=0.45,
    cam_alpha=0.5,
    colormap=cv2.COLORMAP_JET
)

detections = results["detections"]
print(f"Successfully processed {len(detections)} cracks through live pipeline")

session_dict = {
    "uploaded_image": img_np,
    "uploaded_image_name": img_path.name,
    "uploaded_detections": detections,
    "uploaded_prediction": results["annotated"],
    "uploaded_gradcam": results["overlay"],
    "uploaded_confidence": [d["confidence"] for d in detections],
}

metadata = {
    "building_name": "Viaduct Column Pier C-4",
    "location": "Southern Pier Base",
    "inspector": "Eng. P. Sharma, PE",
    "client": "Highways Authority",
}

rep_gen = components["report_generator"]
pkg = rep_gen.generate_full_report_package(session_dict, metadata=metadata)

assert "report_data" in pkg, "Missing report_data"
assert "html_preview" in pkg, "Missing html_preview"
assert "pdf_bytes" in pkg, "Missing pdf_bytes"
assert len(pkg["pdf_bytes"]) > 5000, "PDF too small"
assert pkg["pdf_bytes"][:4] == b"%PDF", "PDF header invalid"
assert "docx_bytes" in pkg, "Missing docx_bytes"
assert len(pkg["docx_bytes"]) > 5000, "DOCX too small"
assert pkg["docx_bytes"][:2] == b"PK", "DOCX header invalid"

rd = pkg["report_data"]
causes = rd["possible_causes"]
assert "Based on the visual characteristics of the detected crack, possible engineering causes may include" in causes, f"Causes phrasing violated: {causes}"
notes = rd["engineering_notes"]
assert "Inspired by terminology and best practices from ACI 224R-01 and Eurocode 2" in notes, f"Notes phrasing violated: {notes}"
assert rd["risk_assessment"]["risk_level"] in ["Low", "Moderate", "High", "Critical"], f"Invalid risk level: {rd['risk_assessment']['risk_level']}"

print("ALL RIGOROUS E2E CHECKS PASSED!")
print(f"PDF generated: {len(pkg['pdf_bytes'])} bytes")
print(f"DOCX generated: {len(pkg['docx_bytes'])} bytes")
print(f"HTML preview: {len(pkg['html_preview'])} bytes")
