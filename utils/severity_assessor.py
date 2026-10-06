"""
Severity Assessor for CrackVision-XAI
Calculates structural crack severity based on physical measurements:
- Crack Length (pixels along skeleton path)
- Average Crack Width (mean aperture)
- Maximum Crack Width (peak aperture)
- Crack Area (actual segmented defect pixels)
- Crack Continuity (connected skeleton ratio)
- Detection Confidence

Classifies into 4 Civil Engineering Structural Risk Tiers:
- Minor (< 30%) -> Green (#10B981)
- Moderate (30% - 55%) -> Orange (#F59E0B)
- Severe (55% - 75%) -> Red (#EF4444)
- Critical (>= 75%) -> Crimson (#991B1B)
"""

import math
from typing import Dict, Tuple, Optional


class SeverityAssessor:
    """
    Evaluates civil structural health risk based on empirical crack geometry,
    aperture metrics, continuity, and model detection certainty.
    """

    COLOR_MAP = {
        "Minor": {
            "hex": "#10B981",          # Emerald Green
            "rgb": (16, 185, 129),
            "bgr": (129, 185, 16),
            "badge_class": "badge-minor"
        },
        "Moderate": {
            "hex": "#F59E0B",          # Amber / Orange
            "rgb": (245, 158, 11),
            "bgr": (11, 158, 245),
            "badge_class": "badge-moderate"
        },
        "Severe": {
            "hex": "#EF4444",          # Vivid Red
            "rgb": (239, 68, 68),
            "bgr": (68, 68, 239),
            "badge_class": "badge-severe"
        },
        "Critical": {
            "hex": "#991B1B",          # Deep Crimson
            "rgb": (153, 27, 27),
            "bgr": (27, 27, 153),
            "badge_class": "badge-critical"
        }
    }

    def __init__(
        self,
        minor_thresh: float = 30.0,
        moderate_thresh: float = 55.0,
        severe_thresh: float = 75.0
    ):
        self.minor_thresh = minor_thresh
        self.moderate_thresh = moderate_thresh
        self.severe_thresh = severe_thresh

    def assess(
        self,
        bbox_w: int,
        bbox_h: int,
        img_w: int,
        img_h: int,
        confidence: float,
        estimated_width_px: Optional[float] = None,
        max_width_px: Optional[float] = None,
        crack_length_px: Optional[float] = None,
        crack_area_px: Optional[int] = None,
        continuity: Optional[float] = None
    ) -> Dict:
        """
        Compute crack severity score and 4-tier structural classification.

        Args:
            bbox_w: Bounding box width in pixels.
            bbox_h: Bounding box height in pixels.
            img_w: Full image width in pixels.
            img_h: Full image height in pixels.
            confidence: Model detection confidence (0.0 - 1.0).
            estimated_width_px: Mean crack aperture in pixels.
            max_width_px: Maximum crack aperture in pixels.
            crack_length_px: Skeleton arc-length in pixels.
            crack_area_px: Actual segmented crack pixels.
            continuity: Connected skeleton component ratio (0.0 - 1.0).

        Returns:
            Dict with severity metrics, 4-tier level, color, and action plan.
        """
        img_area = float(max(1, img_w * img_h))
        img_diag = math.sqrt(img_w**2 + img_h**2)

        # 1. Crack Area (0 - 25 points)
        # Using actual segmented pixel area when available, fallback to bounding area
        if crack_area_px is not None and crack_area_px > 0:
            rel_crack_area = float(crack_area_px) / (img_area * 0.08)
        else:
            rel_crack_area = float(bbox_w * bbox_h) / (img_area * 0.25)
        score_area = min(1.0, rel_crack_area) * 25.0

        # 2. Crack Length (0 - 25 points)
        if crack_length_px is not None and crack_length_px > 0:
            eff_length = float(crack_length_px)
        else:
            eff_length = math.sqrt(bbox_w**2 + bbox_h**2) * 0.9
        score_length = min(1.0, eff_length / (img_diag * 0.65)) * 25.0

        # 3. Average Width (0 - 20 points)
        avg_w = float(estimated_width_px) if estimated_width_px is not None and estimated_width_px > 0 else 2.5
        score_avg_width = min(1.0, avg_w / 10.0) * 20.0

        # 4. Maximum Width (0 - 15 points)
        # Peak aperture indicates local spalling or concrete dilation
        peak_w = float(max_width_px) if max_width_px is not None and max_width_px > 0 else avg_w * 1.5
        score_max_width = min(1.0, peak_w / 18.0) * 15.0

        # 5. Continuity (0 - 8 points)
        cont = float(continuity) if continuity is not None else 0.7
        score_continuity = cont * 8.0

        # 6. Confidence factor (0 - 7 points)
        conf = float(np.clip(confidence, 0.0, 1.0)) if 'np' in globals() else min(1.0, max(0.0, float(confidence)))
        score_conf = conf * 7.0

        # Total continuous score in [0.0, 100.0]
        total_score = score_area + score_length + score_avg_width + score_max_width + score_continuity + score_conf
        severity_score = round(min(100.0, max(0.0, total_score)), 1)

        # 4-Tier Categorization
        if severity_score < self.minor_thresh:
            level = "Minor"
            action = "Superficial hairline defect. Non-structural surface shrinkage crack. Regular periodic photographic monitoring recommended."
            urgency = "Low (Routine Inspection)"
        elif severity_score < self.moderate_thresh:
            level = "Moderate"
            action = "Active surface fracture. Seal fissure with low-viscosity polymer or epoxy injection to prevent water penetration and rebar corrosion."
            urgency = "Medium (Preventive Maintenance)"
        elif severity_score < self.severe_thresh:
            level = "Severe"
            action = "Major structural crack spanning load-bearing member. Conduct non-destructive ultrasonic pulse velocity test and structural load analysis."
            urgency = "High (Engineering Intervention)"
        else:
            level = "Critical"
            action = "Severe structural compromise. Significant aperture width and fissure propagation. Immediate safety shoring, load relief, and structural rehabilitation protocol required."
            urgency = "Critical (Immediate Emergency Action)"

        color_info = self.COLOR_MAP[level]

        return {
            "severity_level": level,
            "severity_score": severity_score,
            "severity_percentage": f"{severity_score:.1f}%",
            "color_hex": color_info["hex"],
            "color_rgb": color_info["rgb"],
            "color_bgr": color_info["bgr"],
            "badge_class": color_info["badge_class"],
            "crack_length_px": round(eff_length, 1),
            "avg_width_px": round(avg_w, 1),
            "max_width_px": round(peak_w, 1),
            "crack_area_px": int(crack_area_px) if crack_area_px is not None else int(bbox_w * bbox_h * 0.15),
            "continuity": round(cont, 2),
            "urgency": urgency,
            "recommended_action": action,
            "component_scores": {
                "area_contribution": round(score_area, 1),
                "length_contribution": round(score_length, 1),
                "avg_width_contribution": round(score_avg_width, 1),
                "max_width_contribution": round(score_max_width, 1),
                "continuity_contribution": round(score_continuity, 1),
                "confidence_contribution": round(score_conf, 1),
            }
        }


if __name__ == "__main__":
    assessor = SeverityAssessor()
    # Test across tiers
    for w, l, mw in [(1.8, 80, 2.5), (4.5, 220, 7.0), (8.5, 450, 14.0), (14.0, 620, 24.0)]:
        res = assessor.assess(
            bbox_w=int(l*0.7), bbox_h=int(l*0.7), img_w=640, img_h=640,
            confidence=0.85, estimated_width_px=w, max_width_px=mw, crack_length_px=l
        )
        print(f"L={l}px, W={w}px -> Severity: {res['severity_level']} ({res['severity_percentage']}), Urgency: {res['urgency']}")
