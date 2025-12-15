"""
YOLO package for object detection and label generation.

This package provides:
- Helper functions for image and mask processing (helpers.py)
- YOLO-specific operations for label generation (yolov8.py)
"""

from .helpers import (
    read_tif,
    mask_to_bbox,
    draw_box_on_image,
    has_mask,
    process_image_with_mask,
    bbox_to_yolo,
)

from .yolov8 import (
    build_yolo_labels_from_masks,
    model,
)

__all__ = [
    # Helper functions
    "read_tif",
    "mask_to_bbox",
    "draw_box_on_image",
    "has_mask",
    "process_image_with_mask",
    "bbox_to_yolo",
    # YOLO functions
    "build_yolo_labels_from_masks",
    "model",
]

