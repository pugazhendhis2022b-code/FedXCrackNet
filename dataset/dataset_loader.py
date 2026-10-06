"""
Dataset Loader for CrackVision-XAI
Automatically parses data.yaml, resolves file paths, validates splits,
and calculates dataset distribution statistics.
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
from PIL import Image


class DatasetLoader:
    """
    Handles discovery, validation, path resolution, and summary
    statistics for YOLOv8 crack detection datasets.
    """

    def __init__(self, data_yaml_path: str = "Concrete Crack.v2i.yolov8/data.yaml"):
        """
        Initialize DatasetLoader with path to data.yaml.
        """
        self.yaml_path = Path(data_yaml_path).resolve()
        if not self.yaml_path.exists():
            # Try searching in current directory
            candidates = list(Path(".").glob("**/data.yaml"))
            if candidates:
                self.yaml_path = candidates[0].resolve()
            else:
                raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")

        self.dataset_root = self.yaml_path.parent
        self.data_config = self._load_yaml()
        self.resolved_paths = self._resolve_split_paths()
        self.class_names = self.data_config.get("names", ["Crack"])
        if isinstance(self.class_names, dict):
            self.class_names = [self.class_names[i] for i in sorted(self.class_names.keys())]

    def _load_yaml(self) -> Dict:
        """Load and parse YAML configuration."""
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _resolve_split_paths(self) -> Dict[str, Dict[str, Path]]:
        """
        Resolve split image and label directory paths, correcting for any
        Roboflow relative path differences.
        """
        splits = {}
        split_keys = {"train": "train", "val": "valid", "test": "test"}

        for split, alias in split_keys.items():
            img_dir = None
            lbl_dir = None

            # Check config path first
            raw_path = self.data_config.get(split)
            if raw_path:
                candidate = (self.yaml_path.parent / raw_path).resolve()
                if candidate.exists():
                    img_dir = candidate

            # Check standard relative subdirectories
            if not img_dir or not img_dir.exists():
                for folder_name in [split, alias]:
                    candidate = self.dataset_root / folder_name / "images"
                    if candidate.exists():
                        img_dir = candidate
                        break

            if img_dir and img_dir.exists():
                lbl_candidate = img_dir.parent / "labels"
                if lbl_candidate.exists():
                    lbl_dir = lbl_candidate

            splits[split] = {"images": img_dir, "labels": lbl_dir}

        return splits

    def get_split_counts(self) -> Dict[str, Dict[str, int]]:
        """
        Return the count of images and label annotations per split.
        """
        counts = {}
        for split, paths in self.resolved_paths.items():
            img_dir = paths.get("images")
            lbl_dir = paths.get("labels")

            img_count = 0
            lbl_count = 0

            if img_dir and img_dir.exists():
                valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
                images = []
                for ext in valid_exts:
                    images.extend(glob.glob(str(img_dir / ext)))
                img_count = len(images)

            if lbl_dir and lbl_dir.exists():
                labels = glob.glob(str(lbl_dir / "*.txt"))
                lbl_count = len(labels)

            counts[split] = {
                "images": img_count,
                "labels": lbl_count,
                "image_dir": str(img_dir) if img_dir else "Not Found",
                "label_dir": str(lbl_dir) if lbl_dir else "Not Found",
            }
        return counts

    def get_summary(self) -> Dict:
        """
        Display and return comprehensive dataset summary:
        Number of training, validation, testing images and class names.
        """
        counts = self.get_split_counts()
        summary = {
            "dataset_name": self.dataset_root.name,
            "yaml_path": str(self.yaml_path),
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "train_images": counts.get("train", {}).get("images", 0),
            "val_images": counts.get("val", {}).get("images", 0),
            "test_images": counts.get("test", {}).get("images", 0),
            "total_images": sum(c.get("images", 0) for c in counts.values()),
            "details": counts,
        }
        return summary

    def print_summary(self):
        """Format and print summary to terminal."""
        s = self.get_summary()
        print("=" * 60)
        print("      CrackVision-XAI: Dataset Loading Summary       ")
        print("=" * 60)
        print(f"Dataset Root       : {s['dataset_name']}")
        print(f"YAML Config Path   : {s['yaml_path']}")
        print(f"Classes ({s['num_classes']})        : {', '.join(s['class_names'])}")
        print("-" * 60)
        print(f"Training Images    : {s['train_images']:,}")
        print(f"Validation Images  : {s['val_images']:,}")
        print(f"Testing Images     : {s['test_images']:,}")
        print(f"Total Images       : {s['total_images']:,}")
        print("=" * 60)

    def get_image_files(self, split: str = "test") -> List[Path]:
        """Get all image file paths for a specified split."""
        img_dir = self.resolved_paths.get(split, {}).get("images")
        if not img_dir or not img_dir.exists():
            return []
        valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        images = []
        for ext in valid_exts:
            images.extend(Path(img_dir).glob(ext))
        return sorted(images)

    def get_sample_images(self, split: str = "test", count: int = 5) -> List[Tuple[Path, Optional[Path]]]:
        """
        Retrieve sample image paths and their corresponding label files.
        """
        images = self.get_image_files(split)
        samples = []
        lbl_dir = self.resolved_paths.get(split, {}).get("labels")

        for img_path in images[:count]:
            lbl_path = None
            if lbl_dir:
                candidate = lbl_dir / f"{img_path.stem}.txt"
                if candidate.exists():
                    lbl_path = candidate
            samples.append((img_path, lbl_path))
        return samples

    def get_normalized_yaml_path(self) -> Path:
        """
        Generate a normalized data.yaml with absolute paths to avoid any
        training path resolution ambiguity with Ultralytics YOLO.
        """
        norm_yaml_path = self.dataset_root / "data_resolved.yaml"
        train_img = self.resolved_paths["train"]["images"]
        val_img = self.resolved_paths["val"]["images"]
        test_img = self.resolved_paths["test"]["images"]

        cfg = {
            "path": str(self.dataset_root).replace("\\", "/"),
            "train": str(train_img.relative_to(self.dataset_root)).replace("\\", "/") if train_img else "train/images",
            "val": str(val_img.relative_to(self.dataset_root)).replace("\\", "/") if val_img else "valid/images",
            "test": str(test_img.relative_to(self.dataset_root)).replace("\\", "/") if test_img else "test/images",
            "nc": len(self.class_names),
            "names": self.class_names,
        }

        with open(norm_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        return norm_yaml_path


if __name__ == "__main__":
    loader = DatasetLoader()
    loader.print_summary()
    norm_yaml = loader.get_normalized_yaml_path()
    print(f"Generated resolved YAML at: {norm_yaml}")
