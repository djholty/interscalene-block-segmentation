"""
Run YOLOv8 inference on DICOM files to generate bounding boxes for region of interest.

This script:
1. Reads DICOM file(s) - supports single file or DICOM series
2. Converts DICOM slices to images
3. Runs YOLOv8 inference on each slice
4. Generates bounding boxes for detected regions
5. Saves visualizations and text files with bounding box information
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

# Try to import DICOM libraries
try:
    import pydicom
    import SimpleITK as sitk
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    print("Warning: pydicom and/or SimpleITK not installed. DICOM support unavailable.")
    print("Install with: pip install pydicom SimpleITK")

# Import directly from helpers file to avoid __init__.py import issues
import importlib.util
helpers_path = Path(__file__).parent / "src" / "yolo" / "helpers.py"
spec = importlib.util.spec_from_file_location("yolo_helpers", helpers_path)
yolo_helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(yolo_helpers)
draw_box_on_image = yolo_helpers.draw_box_on_image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Grayscale -> 3ch conversion (matching YOLOv8 training pipeline)
_CONVERT_LOGGED = {"predict": False}


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
    logger.debug(f"Patched {obj.__class__.__name__}.{method_name} for grayscale->3ch ({tag})")
    return True


def _on_predict_start(predictor):
    """Patch predictor's preprocess method at prediction start."""
    _patch_method(predictor, "preprocess", "predict")


def _register_grayscale_callbacks(model: YOLO) -> None:
    """Register callbacks to patch preprocessing for grayscale->3ch conversion."""
    model.add_callback("on_predict_start", _on_predict_start)
    logger.debug("Registered grayscale->3ch patching callbacks for inference.")


def read_dicom_file(dicom_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Read DICOM file(s) and return as numpy array with metadata.
    
    Args:
        dicom_path: Path to DICOM file or directory containing DICOM series
        
    Returns:
        Tuple of (image_array, metadata_dict)
        - image_array: (N, H, W) for series or (H, W) for single file
        - metadata: Dictionary with DICOM metadata
    """
    if not DICOM_AVAILABLE:
        raise ImportError("DICOM support requires pydicom and SimpleITK")
    
    dicom_path_obj = Path(dicom_path)
    metadata = {}
    
    # Try to read as DICOM series first
    if dicom_path_obj.is_dir():
        try:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_path))
            if dicom_names:
                reader.SetFileNames(dicom_names)
                image_3d = reader.Execute()
                image_array = sitk.GetArrayFromImage(image_3d)
                
                # Get spacing and other metadata
                spacing = image_3d.GetSpacing()
                origin = image_3d.GetOrigin()
                direction = image_3d.GetDirection()
                
                metadata = {
                    'num_slices': len(image_array),
                    'spacing': spacing,
                    'origin': origin,
                    'direction': direction,
                    'type': 'series'
                }
                
                # Normalize to 0-255
                if image_array.max() > 255:
                    image_array = np.clip(image_array, image_array.min(), image_array.max())
                    image_array = ((image_array - image_array.min()) / 
                                 (image_array.max() - image_array.min()) * 255).astype(np.uint8)
                else:
                    image_array = image_array.astype(np.uint8)
                
                logger.info(f"Read DICOM series: {len(image_array)} slices, shape: {image_array.shape}")
                return image_array, metadata
        except Exception as e:
            logger.warning(f"Could not read as DICOM series: {e}")
    
    # Fallback: read single DICOM file
    if dicom_path_obj.is_file():
        ds = pydicom.dcmread(str(dicom_path))
        pixel_array = ds.pixel_array
        
        # Apply windowing if available
        if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            center = float(ds.WindowCenter)
            width = float(ds.WindowWidth)
            pixel_array = np.clip(
                pixel_array,
                center - width / 2,
                center + width / 2
            )
        
        # Normalize to 0-255
        if pixel_array.max() > 255:
            pixel_array = ((pixel_array - pixel_array.min()) / 
                         (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        else:
            pixel_array = pixel_array.astype(np.uint8)
        
        metadata = {
            'num_slices': 1,
            'patient_id': getattr(ds, 'PatientID', 'unknown'),
            'study_date': getattr(ds, 'StudyDate', 'unknown'),
            'type': 'single'
        }
        
        # Return as (1, H, W) for consistency
        if pixel_array.ndim == 2:
            pixel_array = pixel_array[np.newaxis, ...]
        
        logger.info(f"Read single DICOM file: shape {pixel_array.shape}")
        return pixel_array, metadata
    
    raise ValueError(f"Could not read DICOM from: {dicom_path}")


def process_dicom_with_yolo(
    dicom_path: str,
    model_path: str,
    output_dir: str,
    imgsz: int = 384,
    device: Optional[str] = None,
    conf: float = 0.25,
    slice_interval: int = 1,
    max_slices: Optional[int] = None,
) -> Dict:
    """
    Process DICOM file(s) with YOLOv8 and generate bounding boxes.
    
    Args:
        dicom_path: Path to DICOM file or directory
        model_path: Path to YOLOv8 model (.pt file)
        output_dir: Directory to save results
        imgsz: Image size for inference
        device: Device for inference ('cuda', 'cpu', 'mps', or None for auto)
        conf: Confidence threshold
        slice_interval: Process every Nth slice
        max_slices: Maximum number of slices to process
        
    Returns:
        Dictionary with processing statistics
    """
    logger.info("="*60)
    logger.info("YOLOv8 Inference on DICOM Files")
    logger.info("="*60)
    logger.info(f"DICOM path: {dicom_path}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*60)
    
    # Auto-detect device (prioritize MPS on Mac)
    if device is None:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            logger.info("MPS (Metal) available - using Apple GPU acceleration")
        elif torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = YOLO(model_path)
    _register_grayscale_callbacks(model)
    
    # Read DICOM
    image_array, metadata = read_dicom_file(dicom_path)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    visualizations_dir = Path(output_dir) / "visualizations"
    text_output_dir = Path(output_dir) / "bounding_boxes"
    os.makedirs(visualizations_dir, exist_ok=True)
    os.makedirs(text_output_dir, exist_ok=True)
    
    # Get base name for output files
    dicom_name = Path(dicom_path).stem if Path(dicom_path).is_file() else Path(dicom_path).name
    
    # Process slices
    num_slices = len(image_array)
    processed_slices = []
    inference_times = []
    stats = {
        'total_slices': num_slices,
        'processed': 0,
        'detections': 0,
        'no_detections': 0,
    }
    
    logger.info(f"Processing {num_slices} slice(s) from DICOM...")
    
    for slice_idx in range(0, num_slices, slice_interval):
        if max_slices and len(processed_slices) >= max_slices:
            break
        
        try:
            slice_data = image_array[slice_idx]
            
            # Convert to 3-channel for YOLOv8
            if slice_data.ndim == 2:
                slice_3ch = np.stack([slice_data] * 3, axis=-1)
            elif slice_data.ndim == 3 and slice_data.shape[2] == 1:
                slice_3ch = np.repeat(slice_data, 3, axis=2)
            else:
                slice_3ch = slice_data
            
            # Run inference
            start_time = time.perf_counter()
            results = model.predict(
                source=slice_3ch,
                imgsz=imgsz,
                device=device,
                conf=conf,
                verbose=False,
            )
            end_time = time.perf_counter()
            inference_time_ms = (end_time - start_time) * 1000
            inference_times.append(inference_time_ms)
            
            # Process results
            result_image = slice_3ch.copy()
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    stats['detections'] += 1
                    
                    # Get all detections (draw all boxes found)
                    for box_idx in range(len(result.boxes)):
                        box = result.boxes[box_idx]
                        confidence = float(box.conf[0].cpu().numpy())
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        detections.append({
                            'confidence': confidence,
                            'bbox': (x1, y1, x2, y2),
                        })
                        
                        # Draw box on image (use different colors for multiple detections)
                        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                        color = colors[box_idx % len(colors)]
                        result_image = draw_box_on_image(
                            result_image,
                            (x1, y1, x2, y2),
                            color=color,
                            thickness=3,
                        )
                else:
                    stats['no_detections'] += 1
            else:
                stats['no_detections'] += 1
            
            # Save visualization
            vis_path = visualizations_dir / f"{dicom_name}_slice_{slice_idx:06d}_prediction.png"
            Image.fromarray(result_image).save(vis_path)
            
            # Save bounding box information to text file
            txt_path = text_output_dir / f"{dicom_name}_slice_{slice_idx:06d}_bboxes.txt"
            with open(txt_path, 'w') as f:
                f.write(f"Slice: {slice_idx}\n")
                f.write(f"Inference Time: {inference_time_ms:.2f} ms\n")
                f.write(f"Image Dimensions: {slice_data.shape[1]}x{slice_data.shape[0]}\n")
                f.write(f"Number of Detections: {len(detections)}\n")
                f.write("-" * 60 + "\n")
                
                if detections:
                    for i, det in enumerate(detections):
                        f.write(f"\nDetection {i+1}:\n")
                        f.write(f"  Confidence: {det['confidence']:.4f}\n")
                        f.write(f"  Bounding Box (x1, y1, x2, y2): {det['bbox']}\n")
                        x1, y1, x2, y2 = det['bbox']
                        f.write(f"  Width: {x2 - x1} pixels\n")
                        f.write(f"  Height: {y2 - y1} pixels\n")
                else:
                    f.write("No detections found\n")
            
            processed_slices.append({
                'slice_idx': slice_idx,
                'detections': detections,
                'inference_time_ms': inference_time_ms,
            })
            
            stats['processed'] += 1
            
            if stats['processed'] % 10 == 0:
                logger.info(f"Processed {stats['processed']} slices...")
                
        except Exception as e:
            logger.error(f"Error processing slice {slice_idx}: {e}", exc_info=True)
            continue
    
    # Calculate latency statistics
    latency_stats = {}
    if inference_times:
        latency_stats = {
            'min_ms': min(inference_times),
            'max_ms': max(inference_times),
            'avg_ms': sum(inference_times) / len(inference_times),
            'total_inferences': len(inference_times),
        }
    
    # Save summary
    summary_path = Path(output_dir) / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("YOLOv8 DICOM Processing Summary\n")
        f.write("="*60 + "\n")
        f.write(f"DICOM Path: {dicom_path}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Image Size: {imgsz}x{imgsz}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Confidence Threshold: {conf}\n")
        f.write("-"*60 + "\n")
        f.write(f"Total Slices: {stats['total_slices']}\n")
        f.write(f"Processed Slices: {stats['processed']}\n")
        f.write(f"Slices with Detections: {stats['detections']}\n")
        f.write(f"Slices without Detections: {stats['no_detections']}\n")
        f.write("-"*60 + "\n")
        if latency_stats:
            f.write("Inference Latency Statistics:\n")
            f.write(f"  Min: {latency_stats['min_ms']:.2f} ms\n")
            f.write(f"  Max: {latency_stats['max_ms']:.2f} ms\n")
            f.write(f"  Avg: {latency_stats['avg_ms']:.2f} ms\n")
            f.write(f"  Total: {latency_stats['total_inferences']} inferences\n")
            f.write(f"  Approx FPS: {1000 / latency_stats['avg_ms']:.2f}\n")
        f.write("="*60 + "\n")
    
    logger.info("\n" + "="*60)
    logger.info("Processing Complete")
    logger.info("="*60)
    logger.info(f"Processed: {stats['processed']} slices")
    logger.info(f"Detections: {stats['detections']} slices")
    logger.info(f"No detections: {stats['no_detections']} slices")
    logger.info(f"Output directory: {output_dir}")
    
    if latency_stats:
        logger.info("-"*60)
        logger.info("Inference Latency:")
        logger.info(f"  Min: {latency_stats['min_ms']:.2f} ms")
        logger.info(f"  Max: {latency_stats['max_ms']:.2f} ms")
        logger.info(f"  Avg: {latency_stats['avg_ms']:.2f} ms")
        logger.info(f"  FPS: {1000 / latency_stats['avg_ms']:.2f}")
    
    logger.info("="*60)
    
    stats['latency_stats'] = latency_stats
    return stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 inference on DICOM files to generate bounding boxes"
    )
    parser.add_argument(
        "--dicom",
        "-i",
        type=str,
        required=True,
        help="Path to DICOM file or directory containing DICOM series",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="src/yolo/weights/best.pt",
        help="Path to YOLOv8 model (.pt file)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="yolo_dicom_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=384,
        help="Image size for inference (default: 384)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (cuda:0, cpu, mps, or None for auto-detect)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--slice-interval",
        type=int,
        default=1,
        help="Process every Nth slice (default: 1, process all)",
    )
    parser.add_argument(
        "--max-slices",
        type=int,
        default=None,
        help="Maximum number of slices to process (default: all)",
    )
    
    args = parser.parse_args()
    
    if not DICOM_AVAILABLE:
        logger.error("DICOM support requires pydicom and SimpleITK")
        logger.error("Install with: pip install pydicom SimpleITK")
        return 1
    
    try:
        stats = process_dicom_with_yolo(
            dicom_path=args.dicom,
            model_path=args.model,
            output_dir=args.output,
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
            slice_interval=args.slice_interval,
            max_slices=args.max_slices,
        )
        logger.info(f"\n✓ Successfully processed DICOM file(s)")
        logger.info(f"Results saved to: {args.output}")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

