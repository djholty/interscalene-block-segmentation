"""
Helper functions for image and mask processing.

This module contains utility functions for:
- Reading TIF images
- Converting masks to bounding boxes
- Drawing boxes on images
- Processing images with masks
- Converting bounding boxes to YOLO format
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from typing import Optional, Tuple


def read_tif(path: str) -> np.ndarray:
    """
    Read a TIF image file and return as numpy array.
    
    Args:
        path: Path to the TIF file
        
    Returns:
        Image as numpy array
    """
    return np.array(Image.open(path))


def mask_to_bbox(
    mask: np.ndarray,
    pad_px: int = 0,
    pad_frac: float = 0.0,
    square: bool = False,
    target_size: tuple[int, int] | None = None,  # (w, h)
    min_area: int = 1,
) -> tuple[int, int, int, int] | None:
    """
    Convert a mask to a bounding box.
    
    Returns (x1, y1, x2, y2) in pixel coords, inclusive-exclusive style:
    x in [x1, x2), y in [y1, y2)

    Args:
        mask: HxW (or HxWxC) mask. Nonzero is treated as foreground.
        pad_px: Expand bbox by this many pixels each side
        pad_frac: Expand bbox by this fraction of current bbox width/height (each side)
        square: Make the bbox square (keeps center)
        target_size: Force bbox size to (w, h) (keeps center)
        min_area: Ignore tiny masks (in pixels)
        
    Returns:
        Tuple of (x1, y1, x2, y2) or None if mask is too small
    """
    if mask.ndim == 3:
        mask2 = mask[..., 0]
    else:
        mask2 = mask

    fg = mask2 > 0
    if fg.sum() < min_area:
        return None

    ys, xs = np.where(fg)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1

    H, W = mask2.shape[:2]

    # Padding based on fraction of current size
    bw = x2 - x1
    bh = y2 - y1
    frac_pad_x = int(round(pad_frac * bw))
    frac_pad_y = int(round(pad_frac * bh))

    x1 -= (pad_px + frac_pad_x)
    x2 += (pad_px + frac_pad_x)
    y1 -= (pad_px + frac_pad_y)
    y2 += (pad_px + frac_pad_y)

    # Convert to center form for square/target sizing
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = x2 - x1
    bh = y2 - y1

    if square:
        s = max(bw, bh)
        bw = bh = s

    if target_size is not None:
        tw, th = target_size
        bw, bh = tw, th

    # Back to corners
    x1 = int(round(cx - bw / 2.0))
    x2 = int(round(cx + bw / 2.0))
    y1 = int(round(cy - bh / 2.0))
    y2 = int(round(cy + bh / 2.0))

    # Clip to image
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(1, min(W, x2))
    y2 = max(1, min(H, y2))

    # Ensure valid
    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def draw_box_on_image(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    color=(255, 0, 0),
    thickness=3,
) -> np.ndarray:
    """
    Draw a bounding box on an image.
    
    Args:
        image: HxW or HxWx3 image array
        bbox: (x1, y1, x2, y2) bounding box coordinates
        color: RGB color for the box outline
        thickness: Thickness of the box lines
        
    Returns:
        HxWx3 uint8 image with box drawn
    """
    if image.ndim == 2:
        image = np.stack([image]*3, axis=-1)

    img = Image.fromarray(image.astype(np.uint8))
    draw = ImageDraw.Draw(img)

    x1, y1, x2, y2 = bbox
    for t in range(thickness):
        draw.rectangle(
            [x1 - t, y1 - t, x2 + t, y2 + t],
            outline=color
        )

    return np.array(img)


def has_mask(mask: np.ndarray) -> bool:
    """
    Check if mask has any non-zero values (i.e., label present).
    
    Args:
        mask: HxW (or HxWxC) mask array
        
    Returns:
        True if mask has foreground pixels, False otherwise
    """
    if mask.ndim == 3:
        mask2 = mask[..., 0]
    else:
        mask2 = mask
    
    return (mask2 > 0).any()


def process_image_with_mask(
    image_path: str,
    mask_path: str,
    output_path: Optional[str] = None,
    pad_px: int = 0,
    pad_frac: float = 0.0,
    square: bool = False,
    target_size: Optional[Tuple[int, int]] = None,
    min_area: int = 1,
    box_color: Tuple[int, int, int] = (255, 0, 0),
    box_thickness: int = 3,
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """
    Pipeline to read TIF image and mask, create bounding box around mask,
    and optionally save the result with box drawn.
    
    Args:
        image_path: Path to the image TIF file
        mask_path: Path to the mask/label TIF file
        output_path: Optional path to save the output image with box drawn.
                     If None, image is not saved.
        pad_px: Expand bbox by this many pixels each side
        pad_frac: Expand bbox by this fraction of current bbox width/height
        square: Make the bbox square (keeps center)
        target_size: Force bbox size to (w, h) (keeps center)
        min_area: Ignore tiny masks (in pixels)
        box_color: RGB color for the bounding box
        box_thickness: Thickness of the bounding box lines
        
    Returns:
        Tuple of (output_image, bbox) where:
        - output_image: Mask with box drawn if bbox exists, or mask (label) without box if no bbox
        - bbox: (x1, y1, x2, y2) if mask exists and bbox can be created, None otherwise
    """
    # Read image and mask
    image = read_tif(image_path)
    mask = read_tif(mask_path)
    
    # Convert mask to 3-channel if needed for output
    def ensure_3channel(img: np.ndarray) -> np.ndarray:
        """Convert image to 3-channel RGB format."""
        if img.ndim == 2:
            return np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            return np.repeat(img, 3, axis=2)
        return img
    
    mask_output = ensure_3channel(mask.astype(np.uint8))
    
    # Check if mask has any labels
    if not has_mask(mask):
        # No mask present, return mask (label) without box
        if output_path is not None:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            Image.fromarray(mask_output).save(output_path)
        return mask_output, None
    
    # Create bounding box from mask
    bbox = mask_to_bbox(
        mask,
        pad_px=pad_px,
        pad_frac=pad_frac,
        square=square,
        target_size=target_size,
        min_area=min_area,
    )
    
    # If bbox is None (e.g., mask too small), return mask (label) without box
    if bbox is None:
        if output_path is not None:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            Image.fromarray(mask_output).save(output_path)
        return mask_output, None
    
    # Draw box on image
    boxed_image = draw_box_on_image(
        mask,
        bbox,
        color=box_color,
        thickness=box_thickness,
    )
    
    # Optionally save the output
    if output_path is not None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        Image.fromarray(boxed_image).save(output_path)
    
    return boxed_image, bbox


def bbox_to_yolo(
    x1: int, y1: int, x2: int, y2: int, W: int, H: int
) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from pixel coordinates to YOLO format (normalized center coordinates).
    
    Args:
        x1, y1, x2, y2: Bounding box coordinates (x1, y1) top-left, (x2, y2) bottom-right
        W, H: Image width and height
        
    Returns:
        Tuple of (xc, yc, w, h) in normalized YOLO format:
        - xc, yc: Center coordinates normalized to [0, 1]
        - w, h: Width and height normalized to [0, 1]
    """
    xc = ((x1 + x2) / 2.0) / W
    yc = ((y1 + y2) / 2.0) / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    
    # Validate YOLO coordinates are in [0, 1] range
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    
    return xc, yc, w, h

