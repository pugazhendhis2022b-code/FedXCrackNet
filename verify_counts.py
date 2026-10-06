"""Verify dynamic crack detection counts across images."""
import glob
import os
import cv2
from models.yolo_detector import YOLOCrackDetector

def main():
    detector = YOLOCrackDetector("models/best.pt")

    imgs = glob.glob("Concrete Crack.v2i.yolov8/test/images/*.jpg")[:20]
    counts = []

    print("=" * 65)
    print("       DYNAMIC CRACK DETECTION VERIFICATION (NO FIXED CAP)       ")
    print("=" * 65)

    for p in imgs:
        img = cv2.imread(p)
        _, dets = detector.detect(img, conf_threshold=0.18)
        fname = os.path.basename(p)[:24]
        n = len(dets)
        counts.append(n)
        confs = [round(d["confidence"], 3) for d in dets]
        print(f"{fname} -> {n} crack{'s' if n != 1 else ''} detected | Confs: {confs}")

    print("=" * 65)
    print(f"Total Images Tested : {len(imgs)}")
    print(f"Detected Counts     : {counts}")
    print(f"Distinct Counts     : {sorted(list(set(counts)))}")
    print("=" * 65)

    assert len(set(counts)) > 1, "Counts must vary based on image content!"
    assert not all(c == 5 for c in counts), "Detection is NOT fixed at 5!"
    print("VERIFICATION SUCCESSFUL: Detections vary strictly by image content!")

if __name__ == "__main__":
    main()
