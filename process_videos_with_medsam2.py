"""
Pipeline to extract frames from MP4 videos, process them with MedSAM2,
and visualize the resulting masks.

This script:
1. Extracts frames from MP4 files
2. Loads MedSAM2 model with the latest checkpoint
3. Generates masks for each frame using center point prompts
4. Visualizes masks overlaid on original frames
5. Saves visualization results
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import logging

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Add MedSAM2 to path
sys.path.insert(0, str(Path(__file__).parent / "MedSAM2"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_frames_from_video(
    video_path: str, output_dir: str, frame_interval: int = 1
) -> List[str]:
    """
    Extract frames from a video file.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save extracted frames.
        frame_interval (int): Extract every Nth frame (default: 1, extract all frames).

    Returns:
        List[str]: List of paths to extracted frame images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_paths = []
    frame_count = 0
    saved_count = 0
    
    video_name = Path(video_path).stem
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = f"{video_name}_frame_{saved_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    logger.info(
        f"Extracted {len(frame_paths)} frames from {video_path} "
        f"(total frames: {frame_count})"
    )
    return frame_paths


def download_checkpoint_if_needed(checkpoint_path: str) -> bool:
    """
    Download MedSAM2 checkpoint if it doesn't exist.

    Args:
        checkpoint_path (str): Path where checkpoint should be saved.

    Returns:
        bool: True if checkpoint exists or was downloaded successfully, False otherwise.
    """
    if os.path.exists(checkpoint_path):
        return True
    
    logger.info(f"Checkpoint not found at {checkpoint_path}")
    logger.info("Attempting to download MedSAM2_latest.pt...")
    
    try:
        import urllib.request
        
        checkpoint_dir = os.path.dirname(checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        url = "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt"
        
        logger.info(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, checkpoint_path)
        logger.info(f"✓ Successfully downloaded checkpoint to {checkpoint_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download checkpoint: {e}")
        logger.error(
            f"Please download manually using: cd MedSAM2 && bash download.sh"
        )
        return False


def load_medsam2_model(
    checkpoint_path: str,
    config_path: str = "configs/sam2.1_hiera_t512.yaml",
    device: Optional[str] = None,
) -> SAM2ImagePredictor:
    """
    Load MedSAM2 model and create an image predictor.

    Args:
        checkpoint_path (str): Path to the MedSAM2 checkpoint file.
        config_path (str): Path to the model configuration file (relative to sam2 package).
                          Default: "configs/sam2.1_hiera_t512.yaml"
        device (str, optional): Device to use ('cuda', 'mps', 'cpu'). 
                                 If None, auto-detects best available device.

    Returns:
        SAM2ImagePredictor: Initialized MedSAM2 image predictor.
    """
    # Try to download checkpoint if it doesn't exist
    if not os.path.exists(checkpoint_path):
        if not download_checkpoint_if_needed(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Please download it using: cd MedSAM2 && bash download.sh"
            )
    
    logger.info(f"Loading MedSAM2 model from {checkpoint_path}")
    logger.info(f"Using config: {config_path}")
    
    # Build the SAM2 model
    # Note: config_path should be relative to the sam2 package (Hydra searches pkg://sam2)
    sam2_model = build_sam2(
        config_file=config_path,
        ckpt_path=checkpoint_path,
        device=device,
        mode="eval",
    )
    
    # Create image predictor
    predictor = SAM2ImagePredictor(sam2_model)
    
    logger.info(f"Model loaded successfully on device: {predictor.device}")
    return predictor


def generate_mask_with_center_point(
    predictor: SAM2ImagePredictor,
    image: np.ndarray,
    point_coords: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a mask for an image using a center point prompt.

    Args:
        predictor (SAM2ImagePredictor): The MedSAM2 predictor.
        image (np.ndarray): Input image in RGB format (H, W, 3), uint8.
        point_coords (np.ndarray, optional): Point coordinates in (x, y) format.
                                             If None, uses center of image.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of (mask, iou_score).
                                      mask: Binary mask (H, W), bool.
                                      iou_score: IoU prediction score, float.
    """
    # Set image in predictor
    predictor.set_image(image)
    
    # Use center point if not provided
    if point_coords is None:
        h, w = image.shape[:2]
        point_coords = np.array([[w // 2, h // 2]], dtype=np.float32)
    
    # Positive point label (1 = foreground)
    point_labels = np.array([1], dtype=np.int32)
    
    # Predict mask
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,  # Get multiple masks and choose best
    )
    
    # Select mask with highest IoU score
    best_mask_idx = np.argmax(scores)
    best_mask = masks[best_mask_idx] > 0.0
    best_score = scores[best_mask_idx]
    
    return best_mask, best_score


def visualize_mask_on_image(
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


def process_video_frames(
    video_path: str,
    predictor: SAM2ImagePredictor,
    output_dir: str,
    frame_interval: int = 1,
    max_frames: Optional[int] = None,
) -> List[str]:
    """
    Process all frames from a video with MedSAM2 and save visualizations.

    Args:
        video_path (str): Path to input video file.
        predictor (SAM2ImagePredictor): MedSAM2 image predictor.
        output_dir (str): Directory to save output visualizations.
        frame_interval (int): Process every Nth frame.
        max_frames (int, optional): Maximum number of frames to process.

    Returns:
        List[str]: List of paths to saved visualization images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract frames
    frames_dir = os.path.join(output_dir, "frames")
    frame_paths = extract_frames_from_video(
        video_path, frames_dir, frame_interval
    )
    
    if max_frames:
        frame_paths = frame_paths[:max_frames]
    
    video_name = Path(video_path).stem
    visualizations_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)
    
    visualization_paths = []
    
    logger.info(f"Processing {len(frame_paths)} frames from {video_path}")
    
    for frame_path in tqdm(frame_paths, desc="Processing frames"):
        # Load frame
        frame = cv2.imread(frame_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Generate mask
        mask, score = generate_mask_with_center_point(predictor, frame_rgb)
        
        # Visualize
        vis_image = visualize_mask_on_image(frame_rgb, mask)
        
        # Save visualization
        frame_name = Path(frame_path).stem
        vis_path = os.path.join(
            visualizations_dir, f"{video_name}_{frame_name}_masked.jpg"
        )
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(vis_path, vis_image_bgr)
        visualization_paths.append(vis_path)
        
        logger.debug(
            f"Processed {frame_name}: mask area={mask.sum()}, IoU score={score:.3f}"
        )
    
    logger.info(
        f"Saved {len(visualization_paths)} visualizations to {visualizations_dir}"
    )
    return visualization_paths


def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Process MP4 videos with MedSAM2 to generate and visualize masks"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=".",
        help="Directory containing MP4 video files (default: current directory)",
    )
    parser.add_argument(
        "--videos",
        type=str,
        nargs="+",
        default=None,
        help="Specific video files to process (default: all MP4 files in video_dir)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="MedSAM2/checkpoints/MedSAM2_latest.pt",
        help="Path to MedSAM2 checkpoint (default: MedSAM2/checkpoints/MedSAM2_latest.pt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sam2.1_hiera_t512.yaml",
        help="Path to MedSAM2 config file (relative to sam2 package, e.g., 'configs/sam2.1_hiera_t512.yaml')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="medsam2_results",
        help="Directory to save results (default: medsam2_results)",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1, process all frames)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process per video (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to use (default: auto-detect)",
    )
    
    args = parser.parse_args()
    
    # Find video files
    if args.videos:
        video_paths = [os.path.join(args.video_dir, v) for v in args.videos]
    else:
        video_paths = list(Path(args.video_dir).glob("*.mp4"))
        video_paths = [str(p) for p in video_paths]
    
    if not video_paths:
        logger.error(f"No MP4 files found in {args.video_dir}")
        return
    
    logger.info(f"Found {len(video_paths)} video(s) to process")
    
    # Load model once
    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(
            Path(__file__).parent, checkpoint_path
        )
    
    # Config path should be relative to sam2 package (Hydra searches pkg://sam2)
    # Don't convert to absolute path - let Hydra handle it
    config_path = args.config
    
    try:
        predictor = load_medsam2_model(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=args.device,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # Process each video
    for video_path in video_paths:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing video: {video_path}")
        logger.info(f"{'='*60}")
        
        video_name = Path(video_path).stem
        video_output_dir = os.path.join(args.output_dir, video_name)
        
        try:
            visualization_paths = process_video_frames(
                video_path=video_path,
                predictor=predictor,
                output_dir=video_output_dir,
                frame_interval=args.frame_interval,
                max_frames=args.max_frames,
            )
            logger.info(
                f"✓ Successfully processed {video_name}: "
                f"{len(visualization_paths)} frames visualized"
            )
        except Exception as e:
            logger.error(f"✗ Error processing {video_path}: {e}", exc_info=True)
    
    logger.info(f"\n{'='*60}")
    logger.info("Pipeline completed!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

