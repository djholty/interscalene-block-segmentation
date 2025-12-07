"""
Main script for processing medical images and videos with MedSAM2.

Supports multiple input formats:
- MP4/AVI/MOV videos
- DICOM medical images
- TIF/TIFF images (including multi-page)
"""

import argparse
import os
import sys
from pathlib import Path
import logging

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from medsam2_pipeline import MedSAM2Pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Process medical images and videos with MedSAM2 to generate segmentation masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process MP4 videos
  python process_medsam2.py --input MyFile_6.mp4 --output results
  
  # Process DICOM directory
  python process_medsam2.py --input dicom_folder/ --output results
  
  # Process TIF file
  python process_medsam2.py --input image.tif --output results
  
  # Process all supported files in directory
  python process_medsam2.py --input input_dir/ --output results --recursive
  
  # Process with options
  python process_medsam2.py --input video.mp4 --output results --frame_interval 5 --max_images 10
        """
    )
    
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input file or directory to process",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="medsam2_results",
        help="Output directory for results (default: medsam2_results)",
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
        help="Path to MedSAM2 config file (relative to sam2 package)",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=1,
        help="Process every Nth image/frame (default: 1, process all)",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process per file (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process files recursively in directories",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=None,
        help="File extensions to process (default: all supported)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Mask overlay transparency (0.0 to 1.0, default: 0.5)",
    )
    parser.add_argument(
        "--mask_color",
        type=int,
        nargs=3,
        default=[0, 255, 0],
        metavar=("R", "G", "B"),
        help="Mask overlay color in RGB (default: 0 255 0 for green)",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path not found: {args.input}")
        return 1
    
    # Normalize checkpoint path (relative to project root)
    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        # Go up from src/ to project root
        project_root = Path(__file__).parent.parent
        checkpoint_path = str(project_root / checkpoint_path)
    
    # Initialize pipeline
    try:
        pipeline = MedSAM2Pipeline(
            checkpoint_path=checkpoint_path,
            config_path=args.config,
            device=args.device,
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
        return 1
    
    # Process input
    try:
        if input_path.is_file():
            # Single file
            logger.info(f"Processing file: {args.input}")
            result = pipeline.process_file(
                file_path=str(input_path),
                output_dir=args.output,
                frame_interval=args.frame_interval,
                max_images=args.max_images,
                alpha=args.alpha,
                mask_color=tuple(args.mask_color),
            )
            logger.info(f"✓ Successfully processed: {result['num_images']} images")
            
        elif input_path.is_dir():
            # Directory
            logger.info(f"Processing directory: {args.input}")
            results = pipeline.process_directory(
                input_dir=str(input_path),
                output_dir=args.output,
                frame_interval=args.frame_interval,
                max_images=args.max_images,
                file_extensions=args.extensions,
            )
            logger.info(f"✓ Successfully processed {len(results)} file(s)")
            for result in results:
                logger.info(
                    f"  - {Path(result['input_file']).name}: "
                    f"{result['num_images']} images"
                )
        else:
            logger.error(f"Invalid input path: {args.input}")
            return 1
            
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        return 1
    
    logger.info(f"\n{'='*60}")
    logger.info("Pipeline completed successfully!")
    logger.info(f"Results saved to: {args.output}")
    logger.info(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

