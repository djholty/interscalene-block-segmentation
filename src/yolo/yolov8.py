"""
YOLOv8 fine-tuning with optimal strategies for ultrasound images.

This module implements:
- Progressive freezing (freeze backbone, then unfreeze)
- Ultrasound-specific data augmentation
- Optimal hyperparameters for medical imaging
"""

from ultralytics import YOLO
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for ultrasound-specific data augmentation."""
    hsv_s: float = 0.0        # Saturation augmentation
    hsv_v: float = 0.05        # Value (brightness) - important for ultrasound gain variations
    degrees: float = 15.0     # Rotation degrees (±15 degrees)
    translate: float = 0.05     # Translation (5% of image size)
    scale: float = 0.10        # Scale gain (zoom in/out)
    shear: float = 5.0        # Shear degrees (small for medical images)
    perspective: float = 0.0001  # Perspective transform (very small)
    flipud: float = 0.0       # Vertical flip probability (0 = disabled for anatomy)
    fliplr: float = 0.5       # Horizontal flip probability (anatomy can be mirrored)
    mosaic: float = 0.0       # Mosaic augmentation probability
    
    def to_dict(self) -> dict:
        """Convert to dictionary for Ultralytics API."""
        return asdict(self)


@dataclass
class TrainingConfig:
    """Configuration for YOLOv8 training hyperparameters (only valid parameters)."""
    # Learning rate
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    
    # Warmup
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    
    # Loss weights (for object detection)
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    
    # Regularization
    label_smoothing: float = 0.0
    
    # Optimizer and scheduler
    optimizer: str = "auto"
    cos_lr: bool = False
    close_mosaic: int = 10
    
    def to_dict(self) -> dict:
        """Convert to dictionary for Ultralytics API."""
        return asdict(self)


@dataclass
class FreezingConfig:
    """Configuration for progressive freezing strategy."""
    use_progressive_unfreezing: bool = True
    freeze_backbone_epochs: int = 30
    freeze_layers: int = 10  # Number of layers to freeze (backbone) - passed to Ultralytics freeze parameter


def _get_best_checkpoint(results_dir: Path) -> Optional[Path]:
    """Get the best checkpoint path, falling back to last checkpoint if best doesn't exist."""
    best_ckpt = results_dir / "weights" / "best.pt"
    if best_ckpt.exists():
        return best_ckpt
    last_ckpt = results_dir / "weights" / "last.pt"
    return last_ckpt if last_ckpt.exists() else None


def _log_phase_header(title: str, *info_lines: str) -> None:
    """Log a formatted phase header with consistent formatting."""
    logger.info(f"\n{'='*60}")
    logger.info(title)
    for line in info_lines:
        logger.info(line)
    logger.info(f"{'='*60}\n")


def _reduce_augmentation(aug_params: Dict[str, Any], reduction_factor: float = 0.8) -> Dict[str, Any]:
    """Reduce augmentation intensity for fine-tuning phase."""
    keys_to_reduce = {'hsv_v', 'degrees', 'translate'}  # Use set for O(1) lookup
    return {
        k: v * reduction_factor if k in keys_to_reduce else v
        for k, v in aug_params.items()
    }


def _build_training_params(
    base_params: Dict[str, Any],
    train_params: Dict[str, Any],
    aug_params: Dict[str, Any],
    epochs: int,
    name: str,
    freeze: Optional[int] = None,
    lr_override: Optional[float] = None,
    warmup_override: Optional[float] = None,
    pretrained: bool = True,
) -> Dict[str, Any]:
    """Build training parameters dictionary with optional overrides."""
    # Build params with proper override order: base -> train -> aug -> specific overrides
    params = {
        **base_params,
        **train_params,  # Training params can override base
        **aug_params,   # Augmentation params can override training
        'epochs': epochs,
        'name': name,
        'pretrained': pretrained,
    }
    
    # Apply specific overrides (highest priority)
    if freeze is not None:
        params['freeze'] = freeze
    if lr_override is not None:
        params['lr0'] = lr_override
    if warmup_override is not None:
        params['warmup_epochs'] = warmup_override
    
    return params


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
    # Essential training options
    workers: int = 8,
    patience: int = 20,
    verbose: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Train YOLOv8 with optimal fine-tuning strategies for ultrasound images.
    
    This function implements:
    1. Progressive freezing: Freeze backbone initially, then unfreeze
    2. Ultrasound-specific augmentation: Optimized for medical imaging
    3. Optimal hyperparameters for medical imaging
    
    Args:
        data_yaml: Path to dataset/data.yaml file
        model_name: Pretrained checkpoint (e.g., 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt')
        imgsz: Image size (default: 384 for ultrasound)
        epochs: Total number of training epochs
        batch: Batch size
        device: Device string ('cuda:0', 'cpu', '0,1' for multi-GPU)
        project: Project directory name
        name: Run name
        freezing: FreezingConfig object (uses defaults if None)
        training: TrainingConfig object (uses defaults if None)
        augmentation: AugmentationConfig object (uses defaults if None)
        workers: Number of dataloader workers
        patience: Early stopping patience (epochs without improvement)
        verbose: Verbose output
        **kwargs: Additional valid YOLOv8 parameters (e.g., seed, resume, amp, etc.)
        
    Returns:
        Dictionary with training results
        
    Example:
        # Simple usage with defaults
        results = train_yolov8(
            data_yaml="dataset/data.yaml",
            model_name="yolov8n.pt",
            epochs=100
        )
        
        # Custom configuration
        from yolo.yolov8 import AugmentationConfig, TrainingConfig, FreezingConfig
        
        aug = AugmentationConfig(degrees=20.0, translate=0.1)
        train_cfg = TrainingConfig(lr0=0.02, warmup_epochs=5.0)
        freeze_cfg = FreezingConfig(freeze_backbone_epochs=40)
        
        results = train_yolov8(
            data_yaml="dataset/data.yaml",
            augmentation=aug,
            training=train_cfg,
            freezing=freeze_cfg
        )
    """
    # Use defaults if configs not provided
    freezing = freezing or FreezingConfig()
    training = training or TrainingConfig()
    augmentation = augmentation or AugmentationConfig()
    
    # Log training configuration
    logger.info("="*60)
    logger.info("YOLOv8 Fine-tuning for Ultrasound Images")
    logger.info("="*60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Image size: {imgsz}x{imgsz}")
    logger.info(f"Total epochs: {epochs}")
    logger.info(f"Progressive unfreezing: {freezing.use_progressive_unfreezing}")
    if freezing.use_progressive_unfreezing:
        logger.info(f"Freeze backbone epochs: {freezing.freeze_backbone_epochs}")
    logger.info("="*60)
    
    # Load model
    model = YOLO(model_name)
    
    # Convert configs to dictionaries (only once)
    train_params = training.to_dict()
    aug_params = augmentation.to_dict()
    
    # Base training parameters (only valid YOLOv8 parameters)
    base_params = {
        'data': data_yaml,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'project': project,
        'workers': workers,
        'patience': patience,
        'pretrained': True,
        'verbose': verbose,
        **kwargs
    }
    
    # Phase 1: Train with frozen backbone (if progressive unfreezing enabled)
    if freezing.use_progressive_unfreezing and freezing.freeze_backbone_epochs > 0:
        _log_phase_header("Phase 1: Training with Frozen Backbone", 
                         f"Epochs: 1-{freezing.freeze_backbone_epochs}",
                         f"Freezing {freezing.freeze_layers} layers")
        
        # Phase 1: Higher LR for head/neck training with frozen backbone
        phase1_params = _build_training_params(
            base_params=base_params,
            train_params=train_params,
            aug_params=aug_params,
            epochs=freezing.freeze_backbone_epochs,
            name=f"{name}_phase1_frozen",
            freeze=freezing.freeze_layers,
            lr_override=training.lr0 * 1.5,  # Higher LR for head/neck
        )
        
        try:
            results_phase1 = model.train(**phase1_params)
            map50_phase1 = results_phase1.results_dict.get('metrics/mAP50(B)', 'N/A')
            logger.info(f"Phase 1 complete. Best mAP50: {map50_phase1}")
        except Exception as e:
            logger.error(f"Phase 1 training failed: {e}", exc_info=True)
            raise
        
        # Unfreeze backbone for Phase 2
        _log_phase_header("Phase 2: Fine-tuning with Unfrozen Backbone",
                         f"Epochs: {freezing.freeze_backbone_epochs + 1}-{epochs}",
                         "Freezing 0 layers (all layers trainable)")
        
        # Load the best checkpoint from Phase 1
        phase1_ckpt = _get_best_checkpoint(results_phase1.save_dir)
        if phase1_ckpt:
            model = YOLO(str(phase1_ckpt))
            logger.info(f"Loaded Phase 1 checkpoint: {phase1_ckpt}")
        else:
            logger.warning("Phase 1 checkpoint not found, continuing with current model")
        
        # Phase 2: Lower LR for fine-tuning, reduced augmentation, all layers unfrozen
        remaining_epochs = epochs - freezing.freeze_backbone_epochs
        phase2_aug = _reduce_augmentation(aug_params, reduction_factor=0.8)
        
        phase2_params = _build_training_params(
            base_params=base_params,
            train_params=train_params,
            aug_params=phase2_aug,
            epochs=remaining_epochs,
            name=f"{name}_phase2_unfrozen",
            freeze=0,  # Unfreeze all layers
            lr_override=training.lr0 * 0.1,  # Much lower LR for fine-tuning
            warmup_override=training.warmup_epochs * 0.5,  # Shorter warmup
            pretrained=False,  # Continue from Phase 1 checkpoint
        )
        
        try:
            results_phase2 = model.train(**phase2_params)
            map50_phase2 = results_phase2.results_dict.get('metrics/mAP50(B)', 'N/A')
            logger.info(f"Phase 2 complete. Best mAP50: {map50_phase2}")
        except Exception as e:
            logger.error(f"Phase 2 training failed: {e}", exc_info=True)
            raise
        
        logger.info("="*60)
        logger.info("Training Complete!")
        logger.info("="*60)
        
        return {
            'phase1': results_phase1,
            'phase2': results_phase2,
            'final_results': results_phase2
        }
    
    else:
        # Single-phase training (no freezing)
        logger.info("Training without progressive unfreezing")
        
        all_params = _build_training_params(
            base_params=base_params,
            train_params=train_params,
            aug_params=aug_params,
            epochs=epochs,
            name=name,
        )
        
        try:
            results = model.train(**all_params)
            return {'results': results}
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    # Example: Fine-tune YOLOv8 for ultrasound images
    results = train_yolov8(
        data_yaml="src/yolo/dataset/data.yaml",
        model_name="yolov8n.pt",  # or yolov8s.pt, yolov8m.pt for larger models
        imgsz=384,
        epochs=100,
        batch=16,
        device="cuda:0",
        project="runs",
        name="yolo_ultrasound_finetune",
        verbose=True,
    )
    
    print(f"\nTraining complete! Results: {results}")
