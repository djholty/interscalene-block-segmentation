"""
MedSAM2 processor for generating segmentation masks.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Optional
import urllib.request

import numpy as np
import torch

# Add MedSAM2 to path
sys.path.insert(0, str(Path(__file__).parent.parent / "MedSAM2"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

logger = logging.getLogger(__name__)


class MedSAM2Processor:
    """
    Processor for MedSAM2 model operations.
    
    Handles model loading, mask generation, and visualization.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "configs/sam2.1_hiera_t512.yaml",
        device: Optional[str] = None,
    ):
        """
        Initialize MedSAM2 processor.
        
        Args:
            checkpoint_path (str): Path to MedSAM2 checkpoint file.
            config_path (str): Path to model config (relative to sam2 package).
            device (str, optional): Device to use ('cuda', 'mps', 'cpu').
        """
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = device
        self.predictor = None
        self._ensure_checkpoint()
        self._load_model()
    
    def _ensure_checkpoint(self) -> None:
        """Ensure checkpoint exists, download if needed."""
        if os.path.exists(self.checkpoint_path):
            return
        
        logger.info(f"Checkpoint not found at {self.checkpoint_path}")
        logger.info("Attempting to download MedSAM2_latest.pt...")
        
        try:
            checkpoint_dir = os.path.dirname(self.checkpoint_path)
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            url = "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt"
            logger.info(f"Downloading from {url}...")
            urllib.request.urlretrieve(url, self.checkpoint_path)
            logger.info(f"✓ Successfully downloaded checkpoint to {self.checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to download checkpoint: {e}")
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                f"Please download it using: cd MedSAM2 && bash download.sh"
            ) from e
    
    def _load_model(self) -> None:
        """Load MedSAM2 model."""
        logger.info(f"Loading MedSAM2 model from {self.checkpoint_path}")
        logger.info(f"Using config: {self.config_path}")
        
        # Build the SAM2 model
        sam2_model = build_sam2(
            config_file=self.config_path,
            ckpt_path=self.checkpoint_path,
            device=self.device,
            mode="eval",
        )
        
        # Create image predictor
        self.predictor = SAM2ImagePredictor(sam2_model)
        
        logger.info(f"Model loaded successfully on device: {self.predictor.device}")
    
    def generate_mask(
        self,
        image: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        multimask_output: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a mask for an image.
        
        Args:
            image (np.ndarray): Input image in RGB format (H, W, 3), uint8.
            point_coords (np.ndarray, optional): Point coordinates in (x, y) format.
                                                If None, uses center of image.
            multimask_output (bool): If True, returns multiple masks and selects best.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (mask, iou_score)
                                          mask: Binary mask (H, W), bool.
                                          iou_score: IoU prediction score, float.
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Set image in predictor
        self.predictor.set_image(image)
        
        # Use center point if not provided
        if point_coords is None:
            h, w = image.shape[:2]
            point_coords = np.array([[w // 2, h // 2]], dtype=np.float32)
        
        # Positive point label (1 = foreground)
        point_labels = np.array([1], dtype=np.int32)
        
        # Predict mask
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output,
        )
        
        # Select mask with highest IoU score
        if multimask_output and len(scores) > 1:
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx] > 0.0
            best_score = scores[best_mask_idx]
        else:
            best_mask = masks[0] > 0.0
            best_score = scores[0] if len(scores) > 0 else 0.0
        
        return best_mask, best_score
    
    def process_batch(
        self,
        images: list,
        point_coords: Optional[list] = None,
    ) -> list:
        """
        Process a batch of images.
        
        Args:
            images (list): List of images in RGB format (H, W, 3), uint8.
            point_coords (list, optional): List of point coordinate arrays.
        
        Returns:
            list: List of (mask, score) tuples.
        """
        results = []
        for idx, image in enumerate(images):
            coords = point_coords[idx] if point_coords else None
            mask, score = self.generate_mask(image, point_coords=coords)
            results.append((mask, score))
        return results
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.predictor is not None

