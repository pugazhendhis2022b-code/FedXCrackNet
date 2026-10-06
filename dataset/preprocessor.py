"""
Image Preprocessing and Data Augmentation Pipeline for CrackVision-XAI
Implements:
- Image Resizing
- Min-Max and Standard Normalization
- Noise Removal (Bilateral, Gaussian, Median filters)
- Data Augmentation via Albumentations & OpenCV:
  * Horizontal Flip
  * Vertical Flip
  * Rotation
  * Brightness Adjustment
  * Contrast Adjustment
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import albumentations as A


class CrackPreprocessor:
    """
    Comprehensive image preprocessing and augmentation engine for structural crack imagery.
    """

    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        """
        Initialize preprocessor with target resolution.
        """
        self.target_size = target_size
        self.albumentations_pipeline = self._build_augmentation_pipeline()

    def resize(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None,
               interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Resize image to target dimensions.
        """
        size = target_size or self.target_size
        return cv2.resize(image, size, interpolation=interpolation)

    def remove_noise(self, image: np.ndarray, method: str = "bilateral") -> np.ndarray:
        """
        Apply noise removal filtering to suppress concrete surface grain and speckle
        while preserving sharp crack boundaries.

        Methods:
        - 'bilateral': Preserves edges while smoothing background texture (Recommended)
        - 'gaussian': Standard Gaussian smoothing
        - 'median': Non-linear salt-and-pepper noise reduction
        """
        if method == "bilateral":
            # Bilateral filter: diameter=9, sigmaColor=75, sigmaSpace=75
            return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        elif method == "gaussian":
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == "median":
            return cv2.medianBlur(image, 5)
        else:
            return image

    def normalize(self, image: np.ndarray, method: str = "minmax") -> np.ndarray:
        """
        Normalize pixel intensities.

        Methods:
        - 'minmax': Scales values into [0.0, 1.0]
        - 'zscore': Zero-mean unit-variance (standardized)
        - 'clahe': Contrast Limited Adaptive Histogram Equalization for crack contrast
        """
        if method == "minmax":
            return image.astype(np.float32) / 255.0
        elif method == "zscore":
            img_float = image.astype(np.float32)
            mean, std = img_float.mean(), img_float.std() + 1e-7
            return (img_float - mean) / std
        elif method == "clahe":
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                return clahe.apply(image)
        return image

    def _build_augmentation_pipeline(self) -> A.Compose:
        """
        Construct Albumentations augmentation pipeline explicitly covering:
        - Horizontal Flip
        - Vertical Flip
        - Rotation (random rotate 90 + affine rotation)
        - Brightness Adjustment
        - Contrast Adjustment
        """
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=0.7
            ),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        ]

        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.2
            )
        )

    def augment(
        self,
        image: np.ndarray,
        bboxes: Optional[List[List[float]]] = None,
        class_labels: Optional[List[int]] = None
    ) -> Dict[str, Union[np.ndarray, List]]:
        """
        Apply complete data augmentation to image and synchronized YOLO bounding boxes.
        Bboxes should be in YOLO format: [x_center, y_center, width, height] normalized.
        """
        if bboxes is None:
            bboxes = []
        if class_labels is None:
            class_labels = [0] * len(bboxes)

        transformed = self.albumentations_pipeline(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )

        return {
            "image": transformed["image"],
            "bboxes": transformed["bboxes"],
            "class_labels": transformed["class_labels"]
        }

    def generate_augmentation_samples(
        self,
        image: np.ndarray,
        num_variants: int = 4
    ) -> List[Dict[str, Union[str, np.ndarray]]]:
        """
        Generate named individual augmentation variants for demonstration and Streamlit visualization.
        """
        variants = [
            {"title": "Original Image", "image": image.copy()},
            {
                "title": "Bilateral Noise Removal",
                "image": self.remove_noise(image, method="bilateral")
            },
            {
                "title": "Horizontal & Vertical Flip",
                "image": cv2.flip(cv2.flip(image, 1), 0)
            },
            {
                "title": "Rotated (+30°)",
                "image": self._rotate_image(image, angle=30)
            },
            {
                "title": "Enhanced Contrast (CLAHE)",
                "image": self.normalize(image, method="clahe")
            },
            {
                "title": "Albumentations Composite Augmentation",
                "image": self.augment(image)["image"]
            },
        ]
        return variants[:num_variants + 2]

    @staticmethod
    def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by angle in degrees."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rot_mat, (w, h), borderMode=cv2.BORDER_REFLECT_101)


if __name__ == "__main__":
    preprocessor = CrackPreprocessor()
    dummy = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    denoised = preprocessor.remove_noise(dummy, "bilateral")
    augmented = preprocessor.augment(dummy)
    print("Preprocessor initialized and tested successfully.")
    print(f"Original shape: {dummy.shape}, Denoised shape: {denoised.shape}")
    print(f"Augmented shape: {augmented['image'].shape}")
