"""
CrackVision-XAI: One-Command Training Entrypoint
Launches NVIDIA GTX 1650-optimized YOLOv8 training pipeline.

Usage:
    python train_model.py
    python train_model.py --epochs 150 --batch 8
"""

import sys
from pathlib import Path

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from training.train import train_crack_detector, parse_args

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
