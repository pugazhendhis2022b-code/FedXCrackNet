"""Quick end-to-end verification script."""
import cv2
from dataset.dataset_loader import DatasetLoader
from models.yolo_detector import YOLOCrackDetector
from utils.crack_classifier import CrackTypeClassifier
from utils.severity_assessor import SeverityAssessor
from xai.gradcam import YOLOGradCAM

def main():
    loader = DatasetLoader()
    test_images = loader.get_image_files('test')
    print(f"Total test images available: {len(test_images)}")
    test_img_path = test_images[0]
    print(f"Testing pipeline on: {test_img_path.name}")

    img_bgr = cv2.imread(str(test_img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    detector = YOLOCrackDetector('models/best.pt')
    classifier = CrackTypeClassifier()
    severity_assessor = SeverityAssessor()
    gradcam = YOLOGradCAM(detector.model)

    _, dets = detector.detect(img_rgb, conf_threshold=0.05)
    print(f"Detections found: {len(dets)}")

    for d in dets:
        cls_info = classifier.classify(d['crop'], d['width'], d['height'])
        sev_info = severity_assessor.assess(
            bbox_w=d['width'],
            bbox_h=d['height'],
            img_w=img_rgb.shape[1],
            img_h=img_rgb.shape[0],
            confidence=d['confidence'],
            estimated_width_px=cls_info['estimated_width_px']
        )
        print(f"--> Crack #{d['id']}: Conf={d['confidence']:.3f}, Type={cls_info['crack_type']}, "
              f"Severity={sev_info['severity_level']} ({sev_info['severity_percentage']}), "
              f"Angle={cls_info['angle_deg']:.1f}deg")

    hmap = gradcam.generate_heatmap(img_rgb)
    overlay, _ = gradcam.overlay_heatmap(img_rgb, hmap)
    three_p = gradcam.generate_three_panel_view(img_rgb, img_rgb, overlay)
    print(f"3-Panel view assembled with shape: {three_p.shape}")
    print("ALL MODULES VERIFIED SUCCESSFULLY!")

if __name__ == '__main__':
    main()
