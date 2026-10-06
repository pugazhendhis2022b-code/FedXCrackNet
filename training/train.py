"""
CrackVision-XAI: YOLOv8 Training & Optimization Pipeline
Configured for NVIDIA GeForce GTX 1650 (4 GB VRAM) GPU acceleration.
Implements automated dataset configuration, strict CUDA verification, 
hyperparameter tuning, model performance benchmarking, and conditional weight promotion.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
from ultralytics import YOLO

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dataset.dataset_loader import DatasetLoader


def check_cuda_environment(device_id: int = 0) -> str:
    """
    Verify CUDA availability and GPU environment.
    Strictly stops training on CPU to prevent long execution times.
    """
    if not torch.cuda.is_available():
        print("\n" + "=" * 70)
        print("❌ [CUDA ERROR] NVIDIA CUDA GPU is unavailable!")
        print("   Training stopped immediately to prevent slow CPU execution.")
        print("   Environment Check:")
        print(f"   - PyTorch Version: {torch.__version__}")
        print(f"   - CUDA Available : {torch.cuda.is_available()}")
        print("   Please ensure NVIDIA display drivers and a CUDA-enabled PyTorch build are active.")
        print("=" * 70 + "\n")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    if device_id >= device_count:
        print(f"\n❌ [DEVICE ERROR] Requested GPU device '{device_id}', but only {device_count} GPU(s) detected.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(device_id)
    vram_bytes = torch.cuda.get_device_properties(device_id).total_memory
    vram_gb = vram_bytes / (1024 ** 3)

    print("\n" + "=" * 70)
    print("🚀 [CUDA ACTIVE] NVIDIA GPU Acceleration Enabled")
    print(f"   - Device ID      : {device_id}")
    print(f"   - GPU Hardware   : {gpu_name}")
    print(f"   - Total VRAM     : {vram_gb:.2f} GB")
    print(f"   - PyTorch CUDA   : {torch.version.cuda}")
    print("=" * 70)

    return str(device_id)


def evaluate_model_performance(model_path: Path, data_yaml_path: Path, device: str = "0") -> Dict[str, float]:
    """
    Run validation on a model weights file and return Precision, Recall, F1-Score, mAP50, and mAP50-95.
    """
    if not model_path.exists():
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "map50": 0.0,
            "map50_95": 0.0,
        }

    model = YOLO(str(model_path))
    val_results = model.val(
        data=str(data_yaml_path),
        split="val",
        device=device,
        verbose=False,
        plots=False
    )

    p = float(val_results.box.mp)
    r = float(val_results.box.mr)
    f1 = float(2 * p * r / (p + r + 1e-9))
    map50 = float(val_results.box.map50)
    map50_95 = float(val_results.box.map)

    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "map50": map50,
        "map50_95": map50_95,
    }


def compare_and_prompt_promotion(
    current_weights: Path,
    new_weights: Path,
    data_yaml_path: Path,
    device: str = "0",
    auto_replace: bool = False
) -> bool:
    """
    Benchmark new model against current model, display a side-by-side metric comparison,
    and prompt user for confirmation before replacing models/best.pt.
    """
    print("\n" + "=" * 70)
    print("📊 [BENCHMARK] Evaluating Models on Validation Split...")
    print("=" * 70)

    print(f"Evaluating current model: {current_weights.resolve()} ...")
    curr_metrics = evaluate_model_performance(current_weights, data_yaml_path, device=device)

    print(f"Evaluating newly trained model: {new_weights.resolve()} ...")
    new_metrics = evaluate_model_performance(new_weights, data_yaml_path, device=device)

    delta_p = new_metrics["precision"] - curr_metrics["precision"]
    delta_r = new_metrics["recall"] - curr_metrics["recall"]
    delta_f1 = new_metrics["f1"] - curr_metrics["f1"]
    delta_map50 = new_metrics["map50"] - curr_metrics["map50"]
    delta_map = new_metrics["map50_95"] - curr_metrics["map50_95"]

    print("\n" + "=" * 75)
    print("                 MODEL PERFORMANCE COMPARISON TABLE")
    print("=" * 75)
    print(f"{'Metric':<14} | {'Current Model':<16} | {'New Model':<16} | {'Delta':<12} | {'Status'}")
    print("-" * 75)

    metrics_list = [
        ("Precision", curr_metrics["precision"], new_metrics["precision"], delta_p),
        ("Recall", curr_metrics["recall"], new_metrics["recall"], delta_r),
        ("F1-Score", curr_metrics["f1"], new_metrics["f1"], delta_f1),
        ("mAP@50", curr_metrics["map50"], new_metrics["map50"], delta_map50),
        ("mAP@50-95", curr_metrics["map50_95"], new_metrics["map50_95"], delta_map),
    ]

    for name, c_val, n_val, delta in metrics_list:
        status = "▲ Improved" if delta > 0.002 else ("▼ Regressed" if delta < -0.002 else "■ Parity")
        print(f"{name:<14} | {c_val:<16.4f} | {n_val:<16.4f} | {delta:+12.4f} | {status}")
    print("=" * 75)

    # Performance decision criteria: Higher mAP50 or higher F1 with comparable mAP50
    is_better = (new_metrics["map50"] > curr_metrics["map50"]) or (
        abs(delta_map50) < 0.01 and new_metrics["f1"] > curr_metrics["f1"]
    )

    if is_better:
        print("\n🎉 VERDICT: The new model is BETTER than the current model!")
        print(f"   mAP@50 Improvement : {delta_map50 * 100:+.2f}%")
        print(f"   F1-Score Improvement: {delta_f1 * 100:+.2f}%")

        if auto_replace:
            current_weights.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(new_weights, current_weights)
            print(f"\n✅ [AUTO-PROMOTION] models/best.pt successfully updated with new weights.")
            return True

        print("\n" + "-" * 70)
        print("⚠️ PROMOTION CONFIRMATION REQUIRED:")
        print(f"   Source New Model : {new_weights.resolve()}")
        print(f"   Target Location  : {current_weights.resolve()}")
        print("-" * 70)

        try:
            choice = input("Do you want to replace 'models/best.pt' with the new model? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = "n"

        if choice in ["y", "yes"]:
            current_weights.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(new_weights, current_weights)
            print(f"\n✅ [PROMOTED] models/best.pt successfully replaced with new model weights.")
            return True
        else:
            print(f"\nℹ️ [KEPT CURRENT] Replacement skipped by user. models/best.pt remains untouched.")
            print(f"   New weights preserved safely at: {new_weights.resolve()}")
            return False
    else:
        print("\n⚠️ VERDICT: The new model did not outperform the current model.")
        print("   Current model 'models/best.pt' will NOT be replaced.")
        print(f"   New weights remain safely archived at: {new_weights.resolve()}")
        return False


def train_crack_detector(
    data_yaml: Optional[str] = None,
    model_name: str = "yolov8n.pt",
    epochs: int = 150,
    batch_size: int = 2,
    img_size: int = 512,
    device: int = 0,
    workers: int = 2,
    cache: bool = False,
    optimizer: str = "AdamW",
    learning_rate: float = 0.001,
    cos_lr: bool = True,
    close_mosaic: int = 10,
    patience: int = 30,
    amp: bool = False,
    project: str = "runs/detect",
    name: str = "train",
    current_best: str = "models/best.pt",
    auto_replace: bool = False
) -> Path:
    """
    Train YOLOv8n detector with specified hyperparameters on NVIDIA GPU.
    """
    # 1. Enforce CUDA
    active_device = check_cuda_environment(device_id=device)

    # 2. Automatically locate and resolve data.yaml
    loader = DatasetLoader(data_yaml) if data_yaml else DatasetLoader()
    resolved_yaml = loader.get_normalized_yaml_path()
    loader.print_summary()

    # 3. Model initialization
    print(f"\n[Training] Loading base architecture: {model_name}")
    model = YOLO(model_name)

    # 4. Execute Training
    print("\n[Training] Hyperparameter Configuration:")
    print(f"   - Model       : {model_name}")
    print(f"   - Epochs      : {epochs}")
    print(f"   - Batch Size  : {batch_size} (GTX 1650 4GB safe)")
    print(f"   - Image Size  : {img_size}")
    print(f"   - Optimizer   : {optimizer} (lr0={learning_rate}, cos_lr={cos_lr})")
    print(f"   - Workers     : {workers}")
    print(f"   - Cache       : {cache}")
    print(f"   - Close Mosaic: {close_mosaic}")
    print(f"   - Patience    : {patience}")
    print(f"   - Mixed Prec. : AMP={amp}")
    print(f"   - Output Dir  : {project}/{name}")
    print("\nStarting YOLOv8 training...\n")

    train_results = model.train(
        data=str(resolved_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=active_device,
        workers=workers,
        cache=cache,
        optimizer=optimizer,
        lr0=learning_rate,
        cos_lr=cos_lr,
        close_mosaic=close_mosaic,
        patience=patience,
        amp=amp,
        project=project,
        name=name,
        plots=True,
        save=True,
        verbose=True
    )

    # 5. Locate newly trained best weights
    save_dir = Path(train_results.save_dir)
    new_best_weights = save_dir / "weights" / "best.pt"
    print(f"\n[Training Complete] Results saved to: {save_dir.resolve()}")
    print(f"[Training Complete] New best weights: {new_best_weights.resolve()}")

    # 6. Benchmark and compare without automatically overwriting models/best.pt
    target_current = BASE_DIR / current_best
    compare_and_prompt_promotion(
        current_weights=target_current,
        new_weights=new_best_weights,
        data_yaml_path=resolved_yaml,
        device=active_device,
        auto_replace=auto_replace
    )

    return new_best_weights


def parse_args():
    parser = argparse.ArgumentParser(
        description="CrackVision-XAI: YOLOv8 Training Pipeline (NVIDIA GTX 1650 Optimized)"
    )
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml (auto-detected if omitted)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights (default: yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs (default: 150)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8 for 4GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker threads (default: 4)")
    parser.add_argument("--no-cache", action="store_true", help="Disable RAM caching of dataset images")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer (default: AdamW)")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate (default: 0.001)")
    parser.add_argument("--no-cos-lr", action="store_true", help="Disable cosine learning rate schedule")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Epochs to disable mosaic augmentation (default: 10)")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience (default: 30)")
    parser.add_argument("--no-amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)")
    parser.add_argument("--project", type=str, default="runs/detect", help="Output directory (default: runs/detect)")
    parser.add_argument("--name", type=str, default="train", help="Experiment name (default: train)")
    parser.add_argument("--current-best", type=str, default="models/best.pt", help="Current model weights path")
    parser.add_argument("--auto-replace", action="store_true", help="Automatically replace models/best.pt if new model is better")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_crack_detector(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        workers=args.workers,
        cache=not args.no_cache,
        optimizer=args.optimizer,
        learning_rate=args.lr,
        cos_lr=not args.no_cos_lr,
        close_mosaic=args.close_mosaic,
        patience=args.patience,
        amp=not args.no_amp,
        project=args.project,
        name=args.name,
        current_best=args.current_best,
        auto_replace=args.auto_replace
    )
