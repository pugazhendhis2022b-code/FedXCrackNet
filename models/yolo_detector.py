"""
YOLOv8 Crack Detector for CrackVision-XAI
Handles model initialization, real YOLOv8 inference, standard NMS,
and merging of only highly overlapping boxes for the same crack defect.
The number of detections depends entirely on the uploaded image.
"""

import os
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


class YOLOCrackDetector:
    """
    Object-oriented detector interface for YOLOv8 structural crack detection.
    Outputs dynamic crack detections driven purely by the real model inference.
    """

    def __init__(self, model_path: Union[str, Path] = "models/best.pt", device: Optional[str] = None):
        """
        Initialize the YOLOv8 crack detector.
        """
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not self.model_path.exists():
            fallback_weights = Path("yolov8n.pt")
            if fallback_weights.exists():
                print(f"[YOLOCrackDetector] Weights '{self.model_path}' not found. Loading fallback: {fallback_weights}")
                self.model_path = fallback_weights
            else:
                self.model_path = Path("yolov8n.pt")

        self.model = YOLO(str(self.model_path))
        print(f"[YOLOCrackDetector] Model loaded on device: {self.device}")

    @staticmethod
    def _calculate_iou(boxA: List[int], boxB: List[int]) -> float:
        """Compute Intersection over Union between two bounding boxes [x1, y1, x2, y2]."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return float(iou)

    def _merge_highly_overlapping_boxes(self, raw_boxes: List[Dict], iou_thresh: float = 0.50) -> List[Dict]:
        """
        Merge only highly overlapping boxes (IoU >= iou_thresh) belonging to the same physical crack.
        Preserves different physical cracks as separate detections.
        """
        if len(raw_boxes) <= 1:
            return raw_boxes

        # Sort by confidence descending
        boxes = sorted(raw_boxes, key=lambda b: b["confidence"], reverse=True)
        merged = []
        visited = [False] * len(boxes)

        for i in range(len(boxes)):
            if visited[i]:
                continue

            cur_box = list(boxes[i]["box"])
            cur_conf = float(boxes[i]["confidence"])
            visited[i] = True

            for j in range(i + 1, len(boxes)):
                if visited[j]:
                    continue
                iou = self._calculate_iou(cur_box, boxes[j]["box"])
                if iou >= iou_thresh:
                    # Merge redundant box on the same crack
                    cur_box = [
                        min(cur_box[0], boxes[j]["box"][0]),
                        min(cur_box[1], boxes[j]["box"][1]),
                        max(cur_box[2], boxes[j]["box"][2]),
                        max(cur_box[3], boxes[j]["box"][3]),
                    ]
                    cur_conf = max(cur_conf, float(boxes[j]["confidence"]))
                    visited[j] = True

            merged.append({
                "box": cur_box,
                "confidence": cur_conf
            })

        return merged

    def detect(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Perform crack detection on an image.
        The number of detections depends entirely on the uploaded image.
        No artificial slicing, no [:5] limit, no synthetic score scaling.
        """
        # Convert input to RGB numpy array
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Could not load image from {image_input}")
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            rgb_img = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                rgb_img = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                rgb_img = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            else:
                rgb_img = image_input.copy()
        else:
            raise TypeError("Unsupported image input type.")

        img_h, img_w = rgb_img.shape[:2]
        total_pixels = img_h * img_w

        # Run real YOLOv8 prediction with user-specified thresholds
        results = self.model.predict(
            source=rgb_img,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )

        raw_candidates = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = [int(max(0, round(c))) for c in coords]
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)
                conf = float(box.conf[0].cpu().numpy())

                if (x2 - x1) < 2 or (y2 - y1) < 2:
                    continue

                raw_candidates.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": conf
                })

        # Merge only highly overlapping bounding boxes for the same crack
        final_boxes = self._merge_highly_overlapping_boxes(raw_candidates, iou_thresh=0.50)

        # Build dynamic detections list
        detections = []
        for idx, item in enumerate(final_boxes, 1):
            x1, y1, x2, y2 = item["box"]
            conf = item["confidence"]
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            bbox_area = w * h
            rel_area = (bbox_area / total_pixels) * 100.0
            crack_crop = rgb_img[y1:y2, x1:x2].copy()

            detections.append({
                "id": idx,
                "box": [x1, y1, x2, y2],
                "width": w,
                "height": h,
                "area_pixels": bbox_area,
                "rel_area_pct": rel_area,
                "confidence": round(conf, 3),
                "class_id": 0,
                "class_name": "Crack",
                "crop": crack_crop,
            })

        return rgb_img, detections

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        show_type: bool = True,
        show_severity: bool = True
    ) -> np.ndarray:
        """
        Render crisp bounding boxes, confidence badges, crack types,
        and severity status tags onto image.
        """
        canvas = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            conf = det["confidence"]

            color = (37, 99, 235)  # Professional Blue
            if show_severity and "severity_color_bgr" in det:
                color = det["severity_color_bgr"]

            # Draw bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Build label text
            label_parts = [f"Crack #{det['id']}: {conf:.2f}"]
            if show_type and "crack_type" in det:
                label_parts.append(det["crack_type"])
            if show_severity and "severity_level" in det:
                label_parts.append(f"[{det['severity_level']} {det.get('severity_score', 0):.0f}%]")

            label_text = " | ".join(label_parts)

            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
            )
            tag_y1 = max(0, y1 - text_h - 8)
            cv2.rectangle(
                canvas,
                (x1, tag_y1),
                (x1 + text_w + 10, y1),
                color,
                -1
            )
            cv2.putText(
                canvas,
                label_text,
                (x1 + 5, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return canvas


if __name__ == "__main__":
    detector = YOLOCrackDetector("models/best.pt")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    _, dets = detector.detect(dummy, conf_threshold=0.25)
    print(f"Zero detection test on blank surface: {len(dets)} cracks (Expected: 0)")
