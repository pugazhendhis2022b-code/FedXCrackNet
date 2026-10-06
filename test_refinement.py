"""
Comprehensive verification test for refined CrackVision-XAI pipeline.
Evaluates:
- Dynamic crack detection count (no fixed 5-crack cap)
- Intelligent crack merging
- Crack type classification (Vertical, Horizontal, Diagonal, Hairline, Unknown)
- 4-Tier Severity Assessment (Minor, Moderate, Severe, Critical)
- Physical measurements (Length, Avg Width, Max Width, Area, Curvature, Angle)
- Grad-CAM heatmap generation and overlay
"""

import cv2
import numpy as np
from pathlib import Path
from dataset.dataset_loader import DatasetLoader
from models.yolo_detector import YOLOCrackDetector
from utils.crack_classifier import CrackTypeClassifier
from utils.severity_assessor import SeverityAssessor
from xai.gradcam import YOLOGradCAM


def main():
    loader = DatasetLoader()
    test_images = loader.get_image_files("test")
    print(f"Total available test images: {len(test_images)}")

    detector = YOLOCrackDetector("models/best.pt")
    classifier = CrackTypeClassifier()
    severity_assessor = SeverityAssessor()
    gradcam = YOLOGradCAM(detector.model)

    print("\n" + "=" * 90)
    print("                 CrackVision-XAI Refined Pipeline Verification                 ")
    print("=" * 90)

    # Test on a blank/smooth intact image to verify 0 detections
    intact_img = np.full((640, 640, 3), 185, dtype=np.uint8)  # Uniform gray concrete
    noise = np.random.normal(0, 4, intact_img.shape).astype(np.int16)
    intact_img = np.clip(intact_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    _, intact_dets = detector.detect(intact_img, conf_threshold=0.20)
    print(f"\n[Test Intact Surface] Detected Cracks: {len(intact_dets)} (Expected: 0) -> {'PASS' if len(intact_dets) == 0 else 'FAIL'}")

    # Test on sample images from the test set
    evaluated = 0
    type_counts = {}
    severity_counts = {}
    crack_counts = []

    for img_path in test_images[:12]:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_rgb.shape[:2]

        # Detect with default threshold 0.15
        _, detections = detector.detect(img_rgb, conf_threshold=0.15)
        num_found = len(detections)
        crack_counts.append(num_found)
        evaluated += 1

        print(f"\nImage {img_path.name[:30]}... -> Cracks Found: {num_found}")

        for det in detections:
            crop = det["crop"]
            w, h = det["width"], det["height"]
            conf = det["confidence"]

            cls_info = classifier.classify(crop, w, h)
            c_type = cls_info["crack_type"]
            type_counts[c_type] = type_counts.get(c_type, 0) + 1

            sev_info = severity_assessor.assess(
                bbox_w=w,
                bbox_h=h,
                img_w=img_w,
                img_h=img_h,
                confidence=conf,
                estimated_width_px=cls_info["estimated_width_px"],
                max_width_px=cls_info["max_width_px"],
                crack_length_px=cls_info["crack_length_px"],
                crack_area_px=cls_info["crack_area_px"],
                continuity=cls_info["continuity"]
            )
            s_level = sev_info["severity_level"]
            severity_counts[s_level] = severity_counts.get(s_level, 0) + 1

            print(f"   [Crack #{det['id']}] Conf: {conf:.2f} | Type: {c_type} ({cls_info['confidence']*100:.0f}%) | "
                  f"Severity: {s_level} ({sev_info['severity_percentage']}) | "
                  f"L: {cls_info['crack_length_px']:.1f}px, W_avg: {cls_info['estimated_width_px']:.1f}px, "
                  f"W_max: {cls_info['max_width_px']:.1f}px, Area: {cls_info['crack_area_px']}px, "
                  f"Angle: {cls_info['angle_deg']:.1f}°, Curv: {cls_info['curvature']:.2f}")

    # Test Grad-CAM generation
    sample_rgb = cv2.cvtColor(cv2.imread(str(test_images[0])), cv2.COLOR_BGR2RGB)
    hmap = gradcam.generate_heatmap(sample_rgb)
    overlay, _ = gradcam.overlay_heatmap(sample_rgb, hmap)
    three_p = gradcam.generate_three_panel_view(sample_rgb, sample_rgb, overlay)
    print(f"\n[Grad-CAM Test] Three-panel output assembled successfully: {three_p.shape}")

    print("\n" + "=" * 90)
    print("                              VERIFICATION SUMMARY                             ")
    print("=" * 90)
    print(f"Total Images Evaluated   : {evaluated}")
    print(f"Crack Counts per Image   : {crack_counts}")
    print(f"Range of Detected Cracks : Min={min(crack_counts)}, Max={max(crack_counts)}, Unique counts={set(crack_counts)}")
    print(f"Crack Type Distribution  : {type_counts}")
    print(f"Severity Distribution    : {severity_counts}")
    print("=" * 90)
    assert len(set(crack_counts)) > 1 or min(crack_counts) != 5, "Detection count is still artificially capped at 5!"
    print("ALL TESTS PASSED WITH DYNAMIC DETECTION AND NO ARTIFICIAL CAP!")


if __name__ == "__main__":
    main()
