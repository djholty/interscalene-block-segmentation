"""
Main pipeline class for processing medical images and videos with MedSAM2.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import logging

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .input_handlers import BaseInputHandler, get_input_handler
from .processor import MedSAM2Processor

logger = logging.getLogger(__name__)


class MedSAM2Pipeline:
    """
    Main pipeline for processing medical images and videos with MedSAM2.
    
    Supports multiple input formats (MP4, DICOM, TIF) and generates
    segmentation masks with visualizations.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "configs/sam2.1_hiera_t512.yaml",
        device: Optional[str] = None,
    ):
        """
        Initialize the pipeline.
        
        Args:
            checkpoint_path (str): Path to MedSAM2 checkpoint.
            config_path (str): Path to model config (relative to sam2 package).
            device (str, optional): Device to use ('cuda', 'mps', 'cpu').
        """
        self.processor = MedSAM2Processor(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device,
        )
    
    def process_file(
        self,
        file_path: str,
        output_dir: str,
        frame_interval: int = 1,
        max_images: Optional[int] = None,
        alpha: float = 0.5,
        mask_color: Tuple[int, int, int] = (0, 255, 0),
    ) -> dict:
        """
        Process a single file (MP4, DICOM, or TIF).
        
        Args:
            file_path (str): Path to input file.
            output_dir (str): Directory to save results.
            frame_interval (int): Process every Nth image.
            max_images (int, optional): Maximum images to process.
            alpha (float): Mask overlay transparency (0.0 to 1.0).
            mask_color (Tuple[int, int, int]): RGB color for mask overlay.
        
        Returns:
            dict: Results dictionary with keys:
                - 'input_file': Path to input file
                - 'num_images': Number of images processed
                - 'frames_dir': Directory with extracted frames
                - 'visualizations_dir': Directory with mask visualizations
                - 'visualization_paths': List of visualization file paths
        """
        # Get appropriate input handler
        handler = get_input_handler(file_path)
        file_name = Path(file_path).stem
        
        # Create output directories
        frames_dir = os.path.join(output_dir, "frames")
        visualizations_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(visualizations_dir, exist_ok=True)
        
        logger.info(f"Processing {file_path} with {handler.__class__.__name__}")
        
        # Extract images
        images = handler.extract_images(
            file_path=file_path,
            output_dir=frames_dir,
            frame_interval=frame_interval,
            max_images=max_images,
        )
        
        if not images:
            raise ValueError(f"No images extracted from {file_path}")
        
        # Process each image
        visualization_paths = []
        logger.info(f"Generating masks for {len(images)} images...")
        
        for image_path, image_array in tqdm(images, desc="Processing images"):
            # Generate mask
            mask, score = self.processor.generate_mask(image_array)
            
            # Visualize
            vis_image = self._visualize_mask(
                image_array, mask, alpha=alpha, color=mask_color
            )
            
            # Save visualization
            image_name = Path(image_path).stem
            vis_filename = f"{file_name}_{image_name}_masked.jpg"
            vis_path = os.path.join(visualizations_dir, vis_filename)
            vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(vis_path, vis_image_bgr)
            visualization_paths.append(vis_path)
            
            logger.debug(
                f"Processed {image_name}: mask area={mask.sum()}, IoU score={score:.3f}"
            )
        
        logger.info(
            f"✓ Processed {file_name}: {len(visualization_paths)} images visualized"
        )
        
        return {
            'input_file': file_path,
            'num_images': len(images),
            'frames_dir': frames_dir,
            'visualizations_dir': visualizations_dir,
            'visualization_paths': visualization_paths,
        }
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        frame_interval: int = 1,
        max_images: Optional[int] = None,
        file_extensions: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Process all supported files in a directory.
        
        Args:
            input_dir (str): Directory containing input files.
            output_dir (str): Directory to save results.
            frame_interval (int): Process every Nth image.
            max_images (int, optional): Maximum images per file.
            file_extensions (list, optional): File extensions to process.
                                            If None, processes all supported formats.
        
        Returns:
            List[dict]: List of result dictionaries (one per file).
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory not found: {input_dir}")
        
        # Find all supported files
        if file_extensions:
            files = []
            for ext in file_extensions:
                files.extend(input_path.rglob(f"*{ext}"))
        else:
            # Find all potentially supported files
            all_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', 
                            '.dcm', '.dicom', '.tif', '.tiff']
            files = []
            for ext in all_extensions:
                files.extend(input_path.rglob(f"*{ext}"))
        
        # Filter to only supported files
        supported_files = [
            str(f) for f in files
            if get_input_handler(str(f)).supports(str(f))
        ]
        
        if not supported_files:
            logger.warning(f"No supported files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(supported_files)} file(s) to process")
        
        # Process each file
        results = []
        for file_path in supported_files:
            file_name = Path(file_path).stem
            file_output_dir = os.path.join(output_dir, file_name)
            
            try:
                result = self.process_file(
                    file_path=file_path,
                    output_dir=file_output_dir,
                    frame_interval=frame_interval,
                    max_images=max_images,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"✗ Error processing {file_path}: {e}", exc_info=True)
                continue
        
        return results
    
    @staticmethod
    def _visualize_mask(
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """
        Overlay a mask on an image for visualization.
        
        Args:
            image (np.ndarray): Original image in RGB format (H, W, 3), uint8.
            mask (np.ndarray): Binary mask (H, W), bool or uint8.
            alpha (float): Transparency of mask overlay (0.0 to 1.0).
            color (Tuple[int, int, int]): RGB color for mask overlay.
        
        Returns:
            np.ndarray: Image with mask overlaid, RGB format (H, W, 3), uint8.
        """
        # Ensure mask is boolean
        if mask.dtype != bool:
            mask = mask > 0
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = color
        
        # Blend with original image
        result = image.copy()
        overlay_region = mask
        result[overlay_region] = (
            alpha * colored_mask[overlay_region]
            + (1 - alpha) * image[overlay_region]
        ).astype(np.uint8)
        
        return result

