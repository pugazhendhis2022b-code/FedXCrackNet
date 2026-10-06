"""
Crack Type Classifier for CrackVision-XAI
Combines:
- Skeletonization (morphological thinning)
- PCA orientation & eigenvalue ratio
- Contour analysis (area, perimeter, convex hull)
- Aspect ratio of crack bounding profile
- Crack continuity (connected skeleton fraction)
- Crack thickness (mean and max aperture from distance transform)
- Crack curvature (arc-length to chord ratio)

Predicts strictly:
- Vertical Crack
- Horizontal Crack
- Diagonal Crack
- Hairline Crack
- Unknown Crack Type (when confidence is low or crack cannot be reliably classified)
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List


class CrackTypeClassifier:
    """
    Advanced morphological and geometric crack type classifier.
    Extracts physical crack measurements and assigns robust classifications.
    Guarantees reliable classification and strictly returns 'Unknown Crack Type'
    whenever confidence is low or geometry is ambiguous.
    """

    VALID_TYPES = [
        "Vertical Crack",
        "Horizontal Crack",
        "Diagonal Crack",
        "Hairline Crack",
        "Unknown Crack Type"
    ]

    def __init__(
        self,
        hairline_width_thresh: float = 3.6,
        hairline_max_thresh: float = 6.5,
        min_confidence_thresh: float = 0.60
    ):
        self.hairline_width_thresh = hairline_width_thresh
        self.hairline_max_thresh = hairline_max_thresh
        self.min_confidence_thresh = min_confidence_thresh

    @staticmethod
    def extract_crack_mask(crop: np.ndarray) -> np.ndarray:
        """
        Segment concrete crack foreground pixels using CLAHE, Bilateral filtering,
        and Black-Hat morphological transformation to isolate dark fissures from concrete background.
        """
        if crop is None or crop.size == 0:
            return np.zeros((10, 10), dtype=np.uint8)

        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        else:
            gray = crop.copy()

        h, w = gray.shape[:2]

        # Bilateral filter suppresses surface cement paste speckles while preserving crack edges
        smoothed = cv2.bilateralFilter(gray, 5, 45, 45)

        # 1. CLAHE enhanced adaptive thresholding
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(smoothed)
        thresh_ad = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 17, 3
        )

        # 2. Black-Hat transform for narrow dark crevice isolation
        ksize = max(7, min(21, int(min(w, h) * 0.15) | 1))
        bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        blackhat = cv2.morphologyEx(smoothed, cv2.MORPH_BLACKHAT, bh_kernel)
        _, thresh_bh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Fuse adaptive and black-hat masks for superior noise rejection
        if cv2.countNonZero(thresh_bh) >= 12:
            fused = cv2.bitwise_and(thresh_ad, thresh_bh)
            if cv2.countNonZero(fused) < 12:
                fused = thresh_bh
        else:
            fused = thresh_ad

        # Filter small noise dots (< 6 pixels) without eroding thin 1-pixel hairline cracks
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fused, connectivity=8)
        cleaned = np.zeros_like(fused)
        for idx in range(1, num_labels):
            if stats[idx, cv2.CC_STAT_AREA] >= 6:
                cleaned[labels == idx] = 255

        # Fallback to Otsu on enhanced contrast if too sparse
        if cv2.countNonZero(cleaned) < 6:
            _, o_mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(o_mask, connectivity=8)
            cleaned = np.zeros_like(o_mask)
            for idx in range(1, num_labels):
                if stats[idx, cv2.CC_STAT_AREA] >= 4:
                    cleaned[labels == idx] = 255

        return cleaned

    @staticmethod
    def morphological_skeleton(binary_mask: np.ndarray) -> np.ndarray:
        """
        Extract 1-pixel centerline medial skeleton using iterative morphological thinning
        with a cross structuring element.
        """
        if binary_mask is None or binary_mask.size == 0 or cv2.countNonZero(binary_mask) == 0:
            return np.zeros_like(binary_mask)

        skeleton = np.zeros(binary_mask.shape, dtype=np.uint8)
        eroded = binary_mask.copy()
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        for _ in range(60):  # Cap iterations to guarantee termination
            if cv2.countNonZero(eroded) == 0:
                break
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(eroded, opened)
            skeleton = cv2.bitwise_or(skeleton, temp)
            eroded = cv2.erode(eroded, element)

        return skeleton

    def measure_physical_properties(self, crop: np.ndarray, bbox_w: int, bbox_h: int) -> Dict:
        """
        Calculate realistic image-processing-based physical measurements:
        - Crack area (segmented foreground pixels)
        - Skeleton centerline and pixel count
        - Crack length (stereological contour arc-length & skeleton path)
        - Average width (sampled distance transform along medial skeleton)
        - Maximum width (95th percentile distance along medial skeleton)
        - Orientation angle (PCA primary axis on skeleton points)
        - Linearity (ratio of first PCA eigenvalue to total variance)
        - Curvature (arc length / chord distance between endpoints)
        - Continuity (primary component area fraction)
        - Aspect ratio (crop aspect ratio and rotated bounding rect aspect ratio)
        """
        mask = self.extract_crack_mask(crop)
        total_crack_area = int(cv2.countNonZero(mask))

        if total_crack_area < 8:
            diag = float(np.sqrt(bbox_w**2 + bbox_h**2))
            return {
                "crack_area_px": max(total_crack_area, 10),
                "crack_length_px": round(diag * 0.85, 1),
                "avg_width_px": 2.0,
                "max_width_px": 3.0,
                "angle_deg": 45.0,
                "curvature": 1.0,
                "continuity": 0.5,
                "linearity": 0.50,
                "is_reliable": False,
                "mask": mask,
                "skeleton": np.zeros_like(mask)
            }

        # 1. Isolate primary crack component (largest connected component)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            primary_mask = (labels == largest_idx).astype(np.uint8) * 255
            primary_area = int(stats[largest_idx, cv2.CC_STAT_AREA])
            continuity = float(np.clip(primary_area / max(1.0, float(total_crack_area)), 0.1, 1.0))
        else:
            primary_mask = mask
            primary_area = total_crack_area
            continuity = 0.80

        # 2. Skeletonization (Centerline extraction)
        skeleton = self.morphological_skeleton(primary_mask)
        skel_pixels = int(cv2.countNonZero(skeleton))

        # 3. Crack Thickness via Distance Transform along Medial Skeleton
        dist = cv2.distanceTransform(primary_mask, cv2.DIST_L2, 3)
        skel_dist = dist[skeleton > 0] if skel_pixels > 0 else dist[primary_mask > 0]
        if len(skel_dist) > 0:
            avg_width = float(np.mean(skel_dist)) * 2.0
            max_width = float(np.percentile(skel_dist, 95)) * 2.0
        else:
            avg_width = 2.0
            max_width = 3.5

        avg_width = max(1.0, avg_width)
        max_width = max(avg_width, max_width)

        # 4. Contour Analysis, Length & Curvature
        cnts, _ = cv2.findContours(primary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            main_cnt = max(cnts, key=cv2.contourArea)
            peri = float(cv2.arcLength(main_cnt, True))
            skel_len = max(peri / 2.0, float(skel_pixels))
            rect = cv2.minAreaRect(main_cnt)
            rect_w, rect_h = rect[1][0], rect[1][1]
            major_dim = max(rect_w, rect_h)
            minor_dim = min(rect_w, rect_h)
            rot_ar = float(major_dim / max(1.0, minor_dim))
            chord_dist = max(5.0, major_dim)
            curvature = skel_len / max(5.0, chord_dist)
        else:
            skel_len = float(np.sqrt(bbox_w**2 + bbox_h**2) * 0.7)
            chord_dist = skel_len
            curvature = 1.0
            rot_ar = 1.0

        # 5. PCA Orientation and Linearity on Skeleton Points
        pts = np.column_stack(np.where(skeleton > 0)) if skel_pixels >= 6 else np.column_stack(np.where(primary_mask > 0))
        if len(pts) >= 6:
            pts_xy = np.column_stack((pts[:, 1], pts[:, 0])).astype(np.float32)
            try:
                mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts_xy, mean=None)
                dx, dy = float(eigenvectors[0][0]), float(eigenvectors[0][1])
                angle = float(np.degrees(np.arctan2(abs(dy), abs(dx))))
                eig_sum = float(eigenvalues[0] + eigenvalues[1] + 1e-6)
                # Normalized linearity in [0.0, 1.0]: 1.0 is straight line, 0.0 is circle/cross/isotropic
                linearity = float(abs(eigenvalues[0] - eigenvalues[1]) / eig_sum)
            except Exception:
                angle = float(np.degrees(np.arctan2(float(bbox_h), float(max(1, bbox_w)))))
                linearity = 0.65
        else:
            angle = float(np.degrees(np.arctan2(float(bbox_h), float(max(1, bbox_w)))))
            linearity = 0.60

        is_reliable = (
            total_crack_area >= 15 and
            primary_area >= 10 and
            skel_len >= 15.0 and
            linearity >= 0.35 and
            avg_width <= 35.0
        )

        return {
            "crack_area_px": total_crack_area,
            "crack_length_px": round(skel_len, 1),
            "avg_width_px": round(avg_width, 1),
            "max_width_px": round(max_width, 1),
            "angle_deg": round(angle, 1),
            "curvature": round(curvature, 2),
            "continuity": round(continuity, 2),
            "linearity": round(linearity, 2),
            "rot_ar": round(rot_ar, 2),
            "is_reliable": is_reliable,
            "mask": mask,
            "skeleton": skeleton
        }

    def classify(self, crop: np.ndarray, bbox_w: int, bbox_h: int) -> Dict:
        """
        Classify crack using multi-modal physical features.
        Returns:
            crack_type: One of 'Vertical Crack', 'Horizontal Crack', 'Diagonal Crack',
                        'Hairline Crack', or 'Unknown Crack Type'
        """
        metrics = self.measure_physical_properties(crop, bbox_w, bbox_h)

        angle = metrics["angle_deg"]
        avg_w = metrics["avg_width_px"]
        max_w = metrics["max_width_px"]
        skel_len = metrics["crack_length_px"]
        curvature = metrics["curvature"]
        continuity = metrics["continuity"]
        linearity = metrics["linearity"]
        rot_ar = metrics.get("rot_ar", 2.0)
        is_reliable = metrics["is_reliable"]

        aspect_ratio = float(bbox_h) / max(1.0, float(bbox_w))

        # Check 1: Reliability / Ambiguity / Non-elongated defect check -> Unknown Crack Type
        if not is_reliable or linearity < 0.35 or (rot_ar < 1.8 and linearity < 0.50):
            return {
                "crack_type": "Unknown Crack Type",
                "confidence": 0.50,
                "angle_deg": angle,
                "estimated_width_px": avg_w,
                "max_width_px": max_w,
                "crack_area_px": metrics["crack_area_px"],
                "crack_length_px": skel_len,
                "curvature": curvature,
                "continuity": continuity,
                "aspect_ratio": round(aspect_ratio, 2),
                "description": "Fissure geometry is irregular or ambiguous without a dominant axial orientation."
            }

        # Check 2: High non-directional curvature / meandering tortuosity -> Unknown Crack Type
        if curvature > 4.5 and (linearity < 0.75 or rot_ar < 2.0):
            return {
                "crack_type": "Unknown Crack Type",
                "confidence": 0.55,
                "angle_deg": angle,
                "estimated_width_px": avg_w,
                "max_width_px": max_w,
                "crack_area_px": metrics["crack_area_px"],
                "crack_length_px": skel_len,
                "curvature": curvature,
                "continuity": continuity,
                "aspect_ratio": round(aspect_ratio, 2),
                "description": f"High multi-directional tortuosity (index {curvature:.2f}, linearity {linearity*100:.0f}%) prevents single axial orientation classification."
            }

        # Check 3: Orientation vs Bounding Box contradiction -> Unknown Crack Type
        is_contradictory = (
            (angle >= 70.0 and aspect_ratio <= 0.40) or
            (angle <= 20.0 and aspect_ratio >= 2.50)
        )
        if is_contradictory and linearity < 0.80:
            return {
                "crack_type": "Unknown Crack Type",
                "confidence": 0.52,
                "angle_deg": angle,
                "estimated_width_px": avg_w,
                "max_width_px": max_w,
                "crack_area_px": metrics["crack_area_px"],
                "crack_length_px": skel_len,
                "curvature": curvature,
                "continuity": continuity,
                "aspect_ratio": round(aspect_ratio, 2),
                "description": f"Contradictory orientation cues between PCA axis ({angle:.1f}°) and spatial bounding aspect ratio ({aspect_ratio:.2f})."
            }

        # Check 4: Hairline Crack (aperture is narrow, crack is elongated and low curvature)
        is_hairline = (
            avg_w <= self.hairline_width_thresh and
            max_w <= self.hairline_max_thresh and
            (skel_len / max(1.0, avg_w) >= 3.5) and
            curvature <= 3.2 and
            rot_ar >= 2.0
        )
        if is_hairline:
            conf = min(0.96, max(0.75, 0.78 + (1.0 / (avg_w + 0.2)) * 0.15))
            return {
                "crack_type": "Hairline Crack",
                "confidence": round(float(conf), 3),
                "angle_deg": angle,
                "estimated_width_px": avg_w,
                "max_width_px": max_w,
                "crack_area_px": metrics["crack_area_px"],
                "crack_length_px": skel_len,
                "curvature": curvature,
                "continuity": continuity,
                "aspect_ratio": round(aspect_ratio, 2),
                "description": f"Narrow surface micro-fissure (mean aperture: {avg_w:.1f}px, max: {max_w:.1f}px, continuity: {continuity*100:.0f}%)."
            }

        # Check 5: Vertical Crack
        is_vertical = (
            (angle >= 62.0 and aspect_ratio >= 1.05) or
            (angle >= 68.0) or
            (aspect_ratio >= 1.55 and angle >= 48.0)
        )
        if is_vertical and linearity >= 0.40 and curvature <= 4.5:
            raw_conf = min(0.98, 0.78 + (angle / 90.0) * 0.18)
            if linearity < 0.60:
                raw_conf *= (linearity / 0.60)
            if raw_conf >= self.min_confidence_thresh:
                return {
                    "crack_type": "Vertical Crack",
                    "confidence": round(float(raw_conf), 3),
                    "angle_deg": angle,
                    "estimated_width_px": avg_w,
                    "max_width_px": max_w,
                    "crack_area_px": metrics["crack_area_px"],
                    "crack_length_px": skel_len,
                    "curvature": curvature,
                    "continuity": continuity,
                    "aspect_ratio": round(aspect_ratio, 2),
                    "description": f"Vertical orientation ({angle:.1f}° relative to horizon, AR: {aspect_ratio:.2f}, linearity: {linearity*100:.0f}%)."
                }

        # Check 6: Horizontal Crack
        is_horizontal = (
            (angle <= 28.0 and aspect_ratio <= 0.95) or
            (angle <= 22.0) or
            (aspect_ratio <= 0.60 and angle <= 42.0)
        )
        if is_horizontal and linearity >= 0.40 and curvature <= 4.5:
            raw_conf = min(0.98, 0.78 + ((90.0 - angle) / 90.0) * 0.18)
            if linearity < 0.60:
                raw_conf *= (linearity / 0.60)
            if raw_conf >= self.min_confidence_thresh:
                return {
                    "crack_type": "Horizontal Crack",
                    "confidence": round(float(raw_conf), 3),
                    "angle_deg": angle,
                    "estimated_width_px": avg_w,
                    "max_width_px": max_w,
                    "crack_area_px": metrics["crack_area_px"],
                    "crack_length_px": skel_len,
                    "curvature": curvature,
                    "continuity": continuity,
                    "aspect_ratio": round(aspect_ratio, 2),
                    "description": f"Horizontal orientation ({angle:.1f}° relative to horizon, AR: {aspect_ratio:.2f}, linearity: {linearity*100:.0f}%)."
                }

        # Check 7: Diagonal Crack
        is_diagonal = (
            22.0 < angle < 68.0 and
            linearity >= 0.40 and
            curvature <= 4.5
        )
        if is_diagonal:
            dev = abs(angle - 45.0) / 23.0
            raw_conf = min(0.96, 0.80 + (1.0 - min(1.0, dev)) * 0.15)
            if linearity < 0.60:
                raw_conf *= (linearity / 0.60)
            if raw_conf >= self.min_confidence_thresh:
                return {
                    "crack_type": "Diagonal Crack",
                    "confidence": round(float(raw_conf), 3),
                    "angle_deg": angle,
                    "estimated_width_px": avg_w,
                    "max_width_px": max_w,
                    "crack_area_px": metrics["crack_area_px"],
                    "crack_length_px": skel_len,
                    "curvature": curvature,
                    "continuity": continuity,
                    "aspect_ratio": round(aspect_ratio, 2),
                    "description": f"Diagonal shear fracture ({angle:.1f}° inclination, curvature: {curvature:.2f})."
                }

        # Check 8: Low confidence fallback -> Unknown Crack Type
        return {
            "crack_type": "Unknown Crack Type",
            "confidence": 0.52,
            "angle_deg": angle,
            "estimated_width_px": avg_w,
            "max_width_px": max_w,
            "crack_area_px": metrics["crack_area_px"],
            "crack_length_px": skel_len,
            "curvature": curvature,
            "continuity": continuity,
            "aspect_ratio": round(aspect_ratio, 2),
            "description": "Multi-axial dispersion or low classification confidence without a dominant single crack orientation."
        }


if __name__ == "__main__":
    classifier = CrackTypeClassifier()
    # Test with simulated vertical fissure
    strip = np.zeros((140, 50, 3), dtype=np.uint8)
    strip[:, 23:27] = 255
    res = classifier.classify(strip, 50, 140)
    print("Crack Classifier Test:")
    for k, v in res.items():
        print(f"  {k}: {v}")
