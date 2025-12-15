"""
YOLO-related operations for object detection.

This module contains functions for:
- Building YOLO format label files from masks
"""

import os
import glob
import logging
from typing import Optional, Tuple

from helpers import (
    read_tif,
    mask_to_bbox,
    has_mask,
    bbox_to_yolo,
    draw_box_on_image,
)
from PIL import Image
import numpy as np


def build_yolo_labels_from_masks(
    masks_dir: str,
    out_labels_dir: str,
    mask_suffix: str = "_mask.tif",
    class_id: int = 0,
    pad_px: int = 0,
    pad_frac: float = 0.0,
    square: bool = False,
    target_size: Optional[Tuple[int, int]] = None,
        mask_extensions: Tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg"),
        min_area: int = 1,
        create_empty_labels: bool = False,  # Skip masks without visible labels (not used, kept for compatibility)
        verification_images_dir: Optional[str] = None,
    box_color: Tuple[int, int, int] = (255, 0, 0),
    box_thickness: int = 3,
    verbose: bool = True,
) -> dict:
    """
    Build YOLO format label files from mask images.
    
    This function processes mask files directly - no image files are needed.
    For each mask file, it extracts the bounding box and generates a YOLO label file.
    Optionally generates verification images showing the bounding box on the mask.
    
    Args:
        masks_dir: Directory containing mask/label images
        out_labels_dir: Output directory for YOLO label files (.txt)
        mask_suffix: Suffix that identifies mask files (e.g., "_mask.tif")
        class_id: YOLO class ID to use (default: 0)
        pad_px: Expand bbox by this many pixels each side
        pad_frac: Expand bbox by this fraction of current bbox width/height
        square: Make the bbox square (keeps center)
        target_size: Force bbox size to (w, h) (keeps center)
        mask_extensions: Tuple of mask file extensions to process
        min_area: Minimum mask area in pixels to create a bbox
        create_empty_labels: Not used - masks without visible labels are skipped
        verification_images_dir: Optional directory to save verification images with bounding boxes.
                                If None, no verification images are created.
        box_color: RGB color for bounding box visualization (default: red)
        box_thickness: Thickness of bounding box lines in pixels (default: 3)
        verbose: If True, print progress and statistics
        
    Returns:
        Dictionary with statistics:
        - total_masks: Total number of mask files processed
        - labels_created: Number of label files created with bboxes (only masks with visible labels)
        - empty_labels: Always 0 (masks without labels are skipped)
        - skipped: Number of masks skipped (no visible labels or invalid bbox)
        - errors: Number of errors encountered
        - verification_images: Number of verification images created (if verification_images_dir provided)
    """
    logger = logging.getLogger(__name__)
    
    # Validate input directory
    if not os.path.isdir(masks_dir):
        raise ValueError(f"Masks directory does not exist: {masks_dir}")
    
    # Create output directories
    os.makedirs(out_labels_dir, exist_ok=True)
    if verification_images_dir is not None:
        os.makedirs(verification_images_dir, exist_ok=True)
    
    # Find all mask files (files ending with mask_suffix)
    mask_paths = []
    for ext in mask_extensions:
        # Look for files ending with mask_suffix
        pattern = os.path.join(masks_dir, f"*{mask_suffix}")
        mask_paths.extend(glob.glob(pattern))
        # Also try uppercase extensions
        pattern_upper = os.path.join(masks_dir, f"*{mask_suffix.upper()}")
        mask_paths.extend(glob.glob(pattern_upper))
    
    # Remove duplicates and sort
    mask_paths = sorted(set(mask_paths))
    
    if not mask_paths:
        logger.warning(f"No mask files found in {masks_dir} with suffix {mask_suffix}")
        return {
            "total_masks": 0,
            "labels_created": 0,
            "empty_labels": 0,
            "skipped": 0,
            "errors": 0,
            "verification_images": 0,
        }
    
    # Statistics
    stats = {
        "total_masks": len(mask_paths),
        "labels_created": 0,
        "empty_labels": 0,
        "skipped": 0,
        "errors": 0,
        "verification_images": 0,
    }
    
    if verbose:
        logger.info(f"Processing {len(mask_paths)} mask files...")
        logger.info(f"Masks directory: {masks_dir}")
        logger.info(f"Output labels directory: {out_labels_dir}")
        if verification_images_dir:
            logger.info(f"Verification images directory: {verification_images_dir}")
    
    # Process each mask file
    for idx, mask_path in enumerate(mask_paths, 1):
        try:
            # Extract base name by removing mask_suffix
            mask_basename = os.path.basename(mask_path)
            if not mask_basename.endswith(mask_suffix):
                # Try uppercase version
                if mask_basename.endswith(mask_suffix.upper()):
                    base = mask_basename[:-len(mask_suffix.upper())]
                else:
                    # Fallback: remove extension
                    base = os.path.splitext(mask_basename)[0]
            else:
                base = mask_basename[:-len(mask_suffix)]
            
            # Remove extension from base if it still has one
            base = os.path.splitext(base)[0]
            
            label_path = os.path.join(out_labels_dir, base + ".txt")
            
            # Read mask file
            try:
                mask = read_tif(mask_path)
            except Exception as e:
                logger.error(f"Error reading mask {mask_basename}: {e}")
                stats["errors"] += 1
                continue
            
            # Validate mask dimensions
            if mask.ndim < 2:
                logger.warning(f"Invalid mask shape for {base}: {mask.shape}")
                stats["errors"] += 1
                continue
            
            # Get dimensions from mask itself
            H, W = mask.shape[:2]
            
            # Check if mask has any labels - skip masks without visible labels
            if not has_mask(mask):
                stats["skipped"] += 1
                if verbose and idx % 10 == 0:
                    logger.debug(f"  [{idx}/{len(mask_paths)}] {base}: Empty mask, skipped (no visible labels)")
                continue
            
            # Create bounding box from mask
            bbox = mask_to_bbox(
                mask,
                pad_px=pad_px,
                pad_frac=pad_frac,
                square=square,
                target_size=target_size,
                min_area=min_area,
            )
            
            # If bbox is None (mask too small or invalid), skip this mask
            if bbox is None:
                stats["skipped"] += 1
                if verbose and idx % 10 == 0:
                    logger.debug(f"  [{idx}/{len(mask_paths)}] {base}: No valid bbox, skipped")
                continue
            
            # VERIFICATION STEP: Draw bounding box on mask before normalization
            # This allows visual verification that the bbox is correct
            if verification_images_dir is not None:
                try:
                    # Draw box on mask
                    boxed_mask = draw_box_on_image(
                        mask,
                        bbox,
                        color=box_color,
                        thickness=box_thickness,
                    )
                    
                    # Save verification image
                    verification_path = os.path.join(verification_images_dir, f"{base}_bbox_verification.png")
                    Image.fromarray(boxed_mask).save(verification_path)
                    stats["verification_images"] += 1
                    
                    if verbose and idx % 10 == 0:
                        logger.debug(f"  [{idx}/{len(mask_paths)}] {base}: Created verification image")
                except Exception as e:
                    logger.warning(f"Failed to create verification image for {base}: {e}")
            
            # Convert bbox to YOLO format (after verification visualization)
            x1, y1, x2, y2 = bbox
            xc, yc, w, h = bbox_to_yolo(x1, y1, x2, y2, W, H)
            
            # Validate YOLO coordinates
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                logger.warning(f"Invalid YOLO coordinates for {base}: xc={xc:.3f}, yc={yc:.3f}, w={w:.3f}, h={h:.3f}")
                stats["errors"] += 1
                continue
            
            # Write YOLO label file
            with open(label_path, "w") as f:
                f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            
            stats["labels_created"] += 1
            if verbose and idx % 10 == 0:
                logger.debug(f"  [{idx}/{len(mask_paths)}] {base}: Created label with bbox")
                
        except Exception as e:
            logger.error(f"Error processing {mask_path}: {e}", exc_info=True)
            stats["errors"] += 1
            continue
    
    # Print summary
    if verbose:
        logger.info("\n" + "="*60)
        logger.info("YOLO Label Generation Summary")
        logger.info("="*60)
        logger.info(f"Total masks processed: {stats['total_masks']}")
        logger.info(f"Labels created (with visible masks): {stats['labels_created']}")
        logger.info(f"Skipped (no visible labels): {stats['skipped']}")
        logger.info(f"Errors: {stats['errors']}")
        if verification_images_dir:
            logger.info(f"Verification images created: {stats['verification_images']}")
        logger.info("="*60)
    
    return stats


if __name__ == "__main__":
    # Example: Build YOLO labels from masks (no image files needed)
    stats = build_yolo_labels_from_masks(
        masks_dir="data_brachial_plexus",
        out_labels_dir="output/yolo_labels",
        mask_suffix="_mask.tif",
        class_id=0,
        pad_px=15,
        pad_frac=0.10,
        square=True,
        verification_images_dir="output/verification_images",  # Optional: create verification images
        box_color=(255, 0, 0),  # Red bounding boxes
        box_thickness=3,
        verbose=True,
    )
    
    print(f"\nProcessing complete! Stats: {stats}")
