"""
YOLOv8 testing and visualization script.

This module provides:
- Model evaluation on test dataset with metrics
- Visualization of predictions on mask images from data_brachial_plexus
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import logging

from ultralytics import YOLO

from .helpers import read_tif, draw_box_on_image

logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str,
    data_yaml: str,
    imgsz: int = 384,
    device: str | int = "cuda:0",
    conf: float = 0.25,
    iou: float = 0.45,
) -> Dict[str, float]:
    """
    Evaluate YOLOv8 model on test dataset and return metrics.
    
    Args:
        model_path: Path to trained YOLOv8 model (.pt file)
        data_yaml: Path to dataset/data.yaml file
        imgsz: Image size for inference
        device: Device for inference ('cuda:0', 'cpu', etc.)
        conf: Confidence threshold for predictions
        iou: IoU threshold for NMS
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("="*60)
    logger.info("Evaluating YOLOv8 Model")
    logger.info("="*60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Data config: {data_yaml}")
    logger.info(f"Image size: {imgsz}x{imgsz}")
    logger.info(f"Confidence threshold: {conf}")
    logger.info(f"IoU threshold: {iou}")
    logger.info("="*60)
    
    # Load model
    model = YOLO(model_path)
    
    # Run validation on test set
    results = model.val(
        data=data_yaml,
        split='test',
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        verbose=True,
    )
    
    # Extract metrics
    metrics = {
        'mAP50': results.results_dict.get('metrics/mAP50(B)', 0.0),
        'mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0.0),
        'precision': results.results_dict.get('metrics/precision(B)', 0.0),
        'recall': results.results_dict.get('metrics/recall(B)', 0.0),
        'f1': results.results_dict.get('metrics/f1(B)', 0.0),
    }
    
    logger.info("\n" + "="*60)
    logger.info("Evaluation Results")
    logger.info("="*60)
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.4f}")
    logger.info("="*60)
    
    return metrics


def yolo_to_bbox(
    yolo_coords: Tuple[float, float, float, float],
    img_width: int,
    img_height: int,
) -> Tuple[int, int, int, int]:
    """
    Convert YOLO format (normalized center x, center y, width, height) to 
    pixel coordinates (x1, y1, x2, y2).
    
    Args:
        yolo_coords: (x_center, y_center, width, height) normalized [0-1]
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        (x1, y1, x2, y2) in pixel coordinates
    """
    x_center, y_center, width, height = yolo_coords
    
    # Convert to pixel coordinates
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    # Convert to corner coordinates
    x1 = int(x_center_px - width_px / 2)
    y1 = int(y_center_px - height_px / 2)
    x2 = int(x_center_px + width_px / 2)
    y2 = int(y_center_px + height_px / 2)
    
    return x1, y1, x2, y2


def load_yolo_labels(label_path: Path) -> List[Tuple[float, float, float, float]]:
    """
    Load YOLO format labels from a .txt file.
    
    Args:
        label_path: Path to label file
        
    Returns:
        List of (class_id, x_center, y_center, width, height) tuples
    """
    boxes = []
    if not label_path.exists():
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                boxes.append((class_id, x_center, y_center, width, height))
    
    return boxes


def visualize_predictions_on_masks(
    model_path: str,
    test_images_dir: str,
    mask_dir: str,
    output_dir: str,
    imgsz: int = 384,
    device: str | int = "cuda:0",
    conf: float = 0.25,
    box_color: Tuple[int, int, int] = (255, 0, 0),
    box_thickness: int = 3,
    mask_suffix: str = "_mask",
) -> Dict[str, int]:
    """
    Run inference on test images and draw predicted boxes on corresponding mask images.
    
    Args:
        model_path: Path to trained YOLOv8 model (.pt file)
        test_images_dir: Directory containing test images
        mask_dir: Directory containing mask images (data_brachial_plexus)
        output_dir: Directory to save visualization images
        imgsz: Image size for inference
        device: Device for inference
        conf: Confidence threshold for predictions
        box_color: RGB color for bounding boxes
        box_thickness: Thickness of bounding box lines
        mask_suffix: Suffix to identify mask files (e.g., "_mask")
        
    Returns:
        Dictionary with statistics (processed, skipped, errors)
    """
    logger.info("="*60)
    logger.info("Visualizing Predictions on Mask Images")
    logger.info("="*60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Test images: {test_images_dir}")
    logger.info(f"Mask directory: {mask_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*60)
    
    # Load model
    model = YOLO(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get test images
    test_images_path = Path(test_images_dir)
    test_image_files = sorted(test_images_path.glob("*.tif"))
    
    if not test_image_files:
        logger.warning(f"No .tif images found in {test_images_dir}")
        return {'processed': 0, 'skipped': 0, 'errors': 0}
    
    stats = {'processed': 0, 'skipped': 0, 'errors': 0}
    mask_dir_path = Path(mask_dir)
    
    for img_path in test_image_files:
        try:
            # Get base name (e.g., "1_106" from "1_106.tif")
            base_name = img_path.stem
            
            # Find corresponding mask file (always has "mask" in filename)
            # Try with mask suffix first (e.g., "1_106_mask.tif")
            mask_path = mask_dir_path / f"{base_name}{mask_suffix}.tif"
            if not mask_path.exists():
                # Try alternative patterns: mask might be in different positions
                # Look for any file with base_name and "mask" in it
                mask_candidates = list(mask_dir_path.glob(f"*{base_name}*mask*.tif"))
                if not mask_candidates:
                    mask_candidates = list(mask_dir_path.glob(f"*mask*{base_name}*.tif"))
                
                if mask_candidates:
                    mask_path = mask_candidates[0]  # Use first match
                else:
                    logger.warning(f"Mask file with 'mask' in name not found for {base_name}, skipping")
                    stats['skipped'] += 1
                    continue
            
            if not mask_path.exists():
                logger.warning(f"Mask file not found for {base_name}, skipping")
                stats['skipped'] += 1
                continue
            
            # Read mask image
            mask = read_tif(str(mask_path))
            # Ensure mask is 3-channel for visualization
            if mask.ndim == 2:
                mask_3ch = np.stack([mask] * 3, axis=-1).astype(np.uint8)
            elif mask.ndim == 3 and mask.shape[2] == 1:
                mask_3ch = np.repeat(mask, 3, axis=2).astype(np.uint8)
            else:
                mask_3ch = mask.astype(np.uint8)
            
            img_height, img_width = mask.shape[:2]
            
            # Run inference on test image
            results = model.predict(
                source=str(img_path),
                imgsz=imgsz,
                device=device,
                conf=conf,
                verbose=False,
            )
            
            # Draw predicted boxes on mask
            result_image = mask_3ch.copy()
            
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        # Get box coordinates in pixel format (already in original image coordinates)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # YOLOv8 returns coordinates in original image space, but we need to
                        # map them to mask image space. The test image and mask might have different sizes.
                        # Get original test image dimensions
                        test_img = read_tif(str(img_path))
                        test_height, test_width = test_img.shape[:2]
                        
                        # Scale coordinates from test image to mask image
                        if test_height != img_height or test_width != img_width:
                            scale_x = img_width / test_width
                            scale_y = img_height / test_height
                            x1 = int(x1 * scale_x)
                            y1 = int(y1 * scale_y)
                            x2 = int(x2 * scale_x)
                            y2 = int(y2 * scale_y)
                        
                        # Clip to image bounds
                        x1 = max(0, min(img_width - 1, x1))
                        y1 = max(0, min(img_height - 1, y1))
                        x2 = max(1, min(img_width, x2))
                        y2 = max(1, min(img_height, y2))
                        
                        if x2 > x1 and y2 > y1:
                            result_image = draw_box_on_image(
                                result_image,
                                (x1, y1, x2, y2),
                                color=box_color,
                                thickness=box_thickness,
                            )
            
            # Save visualization
            output_path = Path(output_dir) / f"{base_name}_prediction.png"
            Image.fromarray(result_image).save(output_path)
            stats['processed'] += 1
            
            if stats['processed'] % 10 == 0:
                logger.info(f"Processed {stats['processed']} images...")
                
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}", exc_info=True)
            stats['errors'] += 1
    
    logger.info("\n" + "="*60)
    logger.info("Visualization Complete")
    logger.info("="*60)
    logger.info(f"Processed: {stats['processed']}")
    logger.info(f"Skipped: {stats['skipped']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*60)
    
    return stats


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Test YOLOv8 model and visualize predictions on mask images"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained YOLOv8 model (.pt file)"
    )
    parser.add_argument(
        "--data-yaml",
        type=str,
        default="src/yolo/dataset/data.yaml",
        help="Path to dataset/data.yaml file"
    )
    parser.add_argument(
        "--test-images",
        type=str,
        default="src/yolo/dataset/images/test",
        help="Directory containing test images"
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default="data_brachial_plexus",
        help="Directory containing mask images (data_brachial_plexus)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_predictions",
        help="Directory to save visualization images"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=384,
        help="Image size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference (cuda:0, cpu, etc.)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip model evaluation, only generate visualizations"
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualization, only run evaluation"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run evaluation
    if not args.skip_eval:
        try:
            metrics = evaluate_model(
                model_path=args.model,
                data_yaml=args.data_yaml,
                imgsz=args.imgsz,
                device=args.device,
                conf=args.conf,
                iou=args.iou,
            )
            print("\nEvaluation Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            if not args.skip_viz:
                return 1
    
    # Generate visualizations
    if not args.skip_viz:
        try:
            stats = visualize_predictions_on_masks(
                model_path=args.model,
                test_images_dir=args.test_images,
                mask_dir=args.mask_dir,
                output_dir=args.output_dir,
                imgsz=args.imgsz,
                device=args.device,
                conf=args.conf,
            )
            print("\nVisualization Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        except Exception as e:
            logger.error(f"Visualization failed: {e}", exc_info=True)
            return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

