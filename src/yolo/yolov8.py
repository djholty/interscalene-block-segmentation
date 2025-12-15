"""
YOLOv8 fine-tuning with practical strategies for ultrasound images.

Implements:
- Progressive freezing (freeze backbone-ish layers, then unfreeze)
- Ultrasound-friendly augmentation defaults
- Grayscale->3ch conversion by patching preprocess_batch (triplicates 1ch to 3ch)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import logging

import torch
from ultralytics import YOLO
import ultralytics  # for version logging

logger = logging.getLogger(__name__)

# -----------------------------
# Constants for training phases
# -----------------------------
PHASE1_LR_MULTIPLIER = 1.0      # Normal LR for frozen training
PHASE2_LR_MULTIPLIER = 0.01    # Much lower LR (1/100) for fine-tuning unfrozen layers
PHASE2_WARMUP_MULTIPLIER = 2.0  # Longer warmup when unfreezing (double)
AUGMENTATION_REDUCTION_FACTOR = 0.5  # Calmer augmentation in phase 2


# -----------------------------
# Grayscale -> 3ch conversion (triplicate single channel)
# -----------------------------
_CONVERT_LOGGED = {"train": False, "val": False}


def _to_3ch(img: torch.Tensor) -> torch.Tensor:
    """Convert (B,1,H,W)->(B,3,H,W) by triplicating the channel. No-op if already 3ch."""
    if not isinstance(img, torch.Tensor):
        return img
    if img.ndim == 4 and img.shape[1] == 1:
        return img.repeat(1, 3, 1, 1)
    if img.ndim == 3 and img.shape[0] == 1:
        return img.repeat(3, 1, 1)
    return img


def _convert_batch_to_3ch(batch: dict, tag: str) -> dict:
    """Convert batch['img'] from 1ch to 3ch by triplication if needed."""
    if not isinstance(batch, dict) or "img" not in batch:
        return batch
    img = batch["img"]
    new_img = _to_3ch(img)
    if isinstance(img, torch.Tensor) and isinstance(new_img, torch.Tensor):
        if img.shape != new_img.shape and not _CONVERT_LOGGED.get(tag, False):
            logger.info(f"[{tag}] Converted grayscale -> 3ch: {tuple(img.shape)} -> {tuple(new_img.shape)}")
            _CONVERT_LOGGED[tag] = True
    batch["img"] = new_img
    return batch


def _patch_method(obj, method_name: str, tag: str) -> bool:
    """Patch a preprocessing method to convert 1ch->3ch."""
    if obj is None:
        return False
    
    patch_flag = f"_grayscale_patched_{method_name}"
    if getattr(obj, patch_flag, False):
        return False
    if not hasattr(obj, method_name):
        return False
    
    original_method = getattr(obj, method_name)
    
    def patched_method(batch):
        batch = original_method(batch)
        return _convert_batch_to_3ch(batch, tag)
    
    setattr(obj, method_name, patched_method)
    setattr(obj, patch_flag, True)
    logger.info(f"Patched {obj.__class__.__name__}.{method_name} for grayscale->3ch ({tag})")
    return True


def _on_pretrain_routine_start(trainer):
    """Patch trainer's preprocess_batch at the start of training."""
    _patch_method(trainer, "preprocess_batch", "train")


def _on_val_start(validator):
    """Patch validator's preprocess method at validation start."""
    # Validator uses 'preprocess' not 'preprocess_batch'
    _patch_method(validator, "preprocess", "val")


def _register_grayscale_callbacks(model: YOLO) -> None:
    """Register callbacks to patch preprocessing for grayscale->3ch conversion."""
    model.add_callback("on_pretrain_routine_start", _on_pretrain_routine_start)
    model.add_callback("on_val_start", _on_val_start)
    logger.debug("Registered grayscale->3ch patching callbacks.")


# -----------------------------
# Config dataclasses
# -----------------------------
@dataclass
class AugmentationConfig:
    # Ultrasound-safe defaults (conservative)
    hsv_h: float = 0.0
    hsv_s: float = 0.0
    hsv_v: float = 0.05

    degrees: float = 15.0
    translate: float = 0.05
    scale: float = 0.10
    shear: float = 5.0
    perspective: float = 0.0001

    flipud: float = 0.0
    fliplr: float = 0.5

    # Often harmful for medical ultrasound unless carefully validated
    mosaic: float = 0.0
    mixup: float = 0.0
    copy_paste: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainingConfig:
    lr0: float = 0.001         # Lower base LR (let optimizer auto-tune from here)
    lrf: float = 0.01          # Final LR = lr0 * lrf
    momentum: float = 0.937
    weight_decay: float = 0.0005

    warmup_epochs: float = 5.0  # Longer warmup
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1

    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5

    label_smoothing: float = 0.0

    optimizer: str = "AdamW"    # Explicit optimizer (avoid auto which may increase LR)
    cos_lr: bool = True         # Cosine LR decay for smoother training
    close_mosaic: int = 10

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FreezingConfig:
    use_progressive_unfreezing: bool = True
    freeze_backbone_epochs: int = 50   # Longer frozen phase (it was working well)
    freeze_layers: int = 10


# -----------------------------
# Helpers
# -----------------------------
def _get_best_checkpoint(results_dir: Path) -> Optional[Path]:
    best_ckpt = results_dir / "weights" / "best.pt"
    if best_ckpt.exists():
        return best_ckpt
    last_ckpt = results_dir / "weights" / "last.pt"
    return last_ckpt if last_ckpt.exists() else None


def _log_phase_header(title: str, *info_lines: str) -> None:
    logger.info(f"\n{'='*60}")
    logger.info(title)
    for line in info_lines:
        logger.info(line)
    logger.info(f"{'='*60}\n")


def _reduce_augmentation(aug_params: Dict[str, Any], reduction_factor: float = AUGMENTATION_REDUCTION_FACTOR) -> Dict[str, Any]:
    """
    Calm augmentations during fine-tuning. (Ultrasound often benefits from stability.)
    """
    keys_to_reduce = {"hsv_v", "degrees", "translate", "scale", "shear", "perspective"}
    out = {}
    for k, v in aug_params.items():
        if k in keys_to_reduce and isinstance(v, (int, float)):
            out[k] = v * reduction_factor
        else:
            out[k] = v
    return out


def _build_training_params(
    base_params: Dict[str, Any],
    train_params: Dict[str, Any],
    aug_params: Dict[str, Any],
    epochs: int,
    name: str,
    freeze: Optional[int] = None,
    lr_override: Optional[float] = None,
    warmup_override: Optional[float] = None,
    pretrained_flag: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Build Ultralytics YOLO.train kwargs.
    Note: "pretrained" in Ultralytics controls whether to load pretrained weights
    when constructing the model. If we load a checkpoint ourselves, we generally
    don't want Ultralytics to re-initialize anything.
    """
    params = {**base_params, **train_params, **aug_params, "epochs": epochs, "name": name}
    if freeze is not None:
        params["freeze"] = freeze
    if lr_override is not None:
        params["lr0"] = lr_override
    if warmup_override is not None:
        params["warmup_epochs"] = warmup_override
    if pretrained_flag is not None:
        params["pretrained"] = pretrained_flag
    return params


# -----------------------------
# Main training function
# -----------------------------
def train_yolov8(
    data_yaml: str,
    model_name: str = "yolov8n.pt",
    imgsz: int = 384,
    epochs: int = 100,
    batch: int = 16,
    device: str | int = "cuda:0",
    project: str = "runs",
    name: str = "yolo_box_train",
    freezing: Optional[FreezingConfig] = None,
    training: Optional[TrainingConfig] = None,
    augmentation: Optional[AugmentationConfig] = None,
    workers: int = 8,
    patience: int = 30,   # More patience for fine-tuning
    verbose: bool = True,
    seed: Optional[int] = 0,
    **kwargs,
) -> Dict[str, Any]:
    freezing = freezing or FreezingConfig()
    training = training or TrainingConfig()
    augmentation = augmentation or AugmentationConfig()

    logger.info("=" * 60)
    logger.info("YOLOv8 Fine-tuning for Ultrasound Images")
    logger.info("=" * 60)
    logger.info(f"Ultralytics version: {getattr(ultralytics, '__version__', 'unknown')}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Image size: {imgsz}x{imgsz}")
    logger.info(f"Total epochs: {epochs}")
    logger.info(f"Batch: {batch} | Device: {device}")
    logger.info(f"Progressive unfreezing: {freezing.use_progressive_unfreezing}")
    logger.info("=" * 60)

    model = YOLO(model_name)
    _register_grayscale_callbacks(model)

    train_params = training.to_dict()
    aug_params = augmentation.to_dict()

    base_params: Dict[str, Any] = {
        "data": data_yaml,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": project,
        "workers": workers,
        "patience": patience,
        "verbose": verbose,
        "seed": seed,
        **kwargs,
    }

    # Phase training
    if freezing.use_progressive_unfreezing and freezing.freeze_backbone_epochs > 0:
        phase1_epochs = min(epochs, max(1, freezing.freeze_backbone_epochs))

        _log_phase_header(
            "Phase 1: Training with Frozen Early Layers",
            f"Epochs: 1-{phase1_epochs}",
            f"freeze={freezing.freeze_layers} (Ultralytics freezes first N modules; verify for your model variant)",
            f"lr0 = {training.lr0} * {PHASE1_LR_MULTIPLIER}",
        )

        phase1_params = _build_training_params(
            base_params=base_params,
            train_params=train_params,
            aug_params=aug_params,
            epochs=phase1_epochs,
            name=f"{name}_phase1_frozen",
            freeze=freezing.freeze_layers,
            lr_override=training.lr0 * PHASE1_LR_MULTIPLIER,
            # model already loaded from model_name, keep pretrained default behavior
            pretrained_flag=True,
        )
        results_phase1 = model.train(**phase1_params)

        # Load best checkpoint from phase 1
        ckpt = _get_best_checkpoint(Path(results_phase1.save_dir))
        if ckpt is not None:
            model = YOLO(str(ckpt))
            _register_grayscale_callbacks(model)
            logger.info(f"Loaded Phase 1 checkpoint: {ckpt}")
        else:
            logger.warning("Phase 1 checkpoint not found; continuing with current model weights.")

        remaining_epochs = max(1, epochs - phase1_epochs)
        phase2_aug = _reduce_augmentation(aug_params, reduction_factor=AUGMENTATION_REDUCTION_FACTOR)

        _log_phase_header(
            "Phase 2: Fine-tuning with All Layers Trainable",
            f"Epochs: {phase1_epochs + 1}-{phase1_epochs + remaining_epochs}",
            "freeze=0 (all trainable)",
            f"lr0 = {training.lr0} * {PHASE2_LR_MULTIPLIER}",
            f"warmup_epochs = {training.warmup_epochs} * {PHASE2_WARMUP_MULTIPLIER}",
            f"Aug reduction factor: {AUGMENTATION_REDUCTION_FACTOR}",
        )

        phase2_params = _build_training_params(
            base_params=base_params,
            train_params=train_params,
            aug_params=phase2_aug,
            epochs=remaining_epochs,
            name=f"{name}_phase2_unfrozen",
            freeze=0,
            lr_override=training.lr0 * PHASE2_LR_MULTIPLIER,
            warmup_override=training.warmup_epochs * PHASE2_WARMUP_MULTIPLIER,
            # We loaded a checkpoint explicitly; don't ask ultralytics to treat it as "pretrained init"
            pretrained_flag=False,
        )
        results_phase2 = model.train(**phase2_params)

        return {"phase1": results_phase1, "phase2": results_phase2, "final_results": results_phase2}

    # Single-phase training
    _log_phase_header("Single Phase: Training", f"Epochs: 1-{epochs}")

    all_params = _build_training_params(
        base_params=base_params,
        train_params=train_params,
        aug_params=aug_params,
        epochs=epochs,
        name=name,
        pretrained_flag=True,
    )
    results = model.train(**all_params)
    return {"results": results}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    results = train_yolov8(
        data_yaml="src/yolo/dataset/data.yaml",
        model_name="yolov8n.pt",
        imgsz=384,
        epochs=120,      # 50 frozen + 70 unfrozen
        batch=16,
        device="cuda:0",
        project="runs",
        name="yolo_ultrasound_finetune",
        patience=30,     # More patience for fine-tuning recovery
        verbose=True,
    )

    print(f"\nTraining complete! Results keys: {list(results.keys())}")
