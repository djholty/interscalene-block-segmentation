# MedSAM2 Video Processing Pipeline

This pipeline extracts frames from MP4 videos, processes them with MedSAM2 to generate segmentation masks, and visualizes the results.

## Features

- **Frame Extraction**: Extracts frames from MP4 video files
- **MedSAM2 Integration**: Uses the latest MedSAM2 checkpoint for medical image segmentation
- **Automatic Mask Generation**: Generates masks using center point prompts
- **Visualization**: Overlays masks on original frames for easy inspection
- **Batch Processing**: Processes multiple videos automatically

## Requirements

- Python 3.8+
- PyTorch
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- MedSAM2 dependencies (see MedSAM2/README.md)

## Installation

1. Install dependencies:
```bash
pip install opencv-python numpy matplotlib tqdm
```

2. Ensure MedSAM2 is set up (see `MedSAM2/README.md`)

3. Download MedSAM2 checkpoint (if not already present):
```bash
cd MedSAM2 && bash download.sh
```

Or the script will attempt to download it automatically if missing.

## Usage

### Basic Usage

Process all MP4 files in the current directory:
```bash
python process_videos_with_medsam2.py
```

### Process Specific Videos

```bash
python process_videos_with_medsam2.py --videos MyFile_6.mp4 MyFile_7.mp4
```

### Process Videos from a Directory

```bash
python process_videos_with_medsam2.py --video_dir /path/to/videos
```

### Process Every Nth Frame

```bash
python process_videos_with_medsam2.py --frame_interval 5
```

### Limit Number of Frames

```bash
python process_videos_with_medsam2.py --max_frames 10
```

### Specify Output Directory

```bash
python process_videos_with_medsam2.py --output_dir my_results
```

### Use Specific Device

```bash
python process_videos_with_medsam2.py --device cuda
```

## Command Line Arguments

- `--video_dir`: Directory containing MP4 files (default: current directory)
- `--videos`: Specific video files to process (default: all MP4 files)
- `--checkpoint`: Path to MedSAM2 checkpoint (default: `MedSAM2/checkpoints/MedSAM2_latest.pt`)
- `--config`: Path to MedSAM2 config file relative to sam2 package (default: `configs/sam2.1_hiera_t512.yaml`)
  
  **Note**: The config path should be relative to the `sam2` package, not an absolute path. Hydra searches for configs in `pkg://sam2`, so use paths like `configs/sam2.1_hiera_t512.yaml`.
- `--output_dir`: Directory to save results (default: `medsam2_results`)
- `--frame_interval`: Process every Nth frame (default: 1, all frames)
- `--max_frames`: Maximum frames per video (default: all)
- `--device`: Device to use: `cuda`, `mps`, or `cpu` (default: auto-detect)

## Output Structure

Results are saved in the following structure:
```
output_dir/
├── video_name_1/
│   ├── frames/              # Extracted frames
│   │   ├── video_name_1_frame_000000.jpg
│   │   └── ...
│   └── visualizations/      # Mask visualizations
│       ├── video_name_1_frame_000000_masked.jpg
│       └── ...
└── video_name_2/
    └── ...
```

## How It Works

1. **Frame Extraction**: Uses OpenCV to extract frames from MP4 files
2. **Model Loading**: Loads MedSAM2 model with the latest checkpoint
3. **Mask Generation**: For each frame:
   - Uses center point of image as prompt
   - Generates multiple masks and selects the best one (highest IoU score)
4. **Visualization**: Overlays the mask on the original frame with transparency
5. **Saving**: Saves both extracted frames and visualizations

## Testing

Run unit tests:
```bash
pytest tests/test_video_processing.py -v
```

## Troubleshooting

### Checkpoint Not Found

If you see an error about missing checkpoint:
1. Run `cd MedSAM2 && bash download.sh` to download all checkpoints
2. Or the script will attempt to download `MedSAM2_latest.pt` automatically

### Out of Memory

If you run out of memory:
- Use `--frame_interval` to process fewer frames
- Use `--max_frames` to limit frames per video
- Process videos one at a time

### Device Issues

- CUDA: Ensure PyTorch with CUDA support is installed
- MPS: Available on Apple Silicon Macs
- CPU: Will work but may be slow

## Example

```bash
# Process all MP4 files, every 5th frame, max 20 frames per video
python process_videos_with_medsam2.py \
    --frame_interval 5 \
    --max_frames 20 \
    --output_dir results
```

This will:
1. Find all `.mp4` files in the current directory
2. Extract every 5th frame from each video
3. Process up to 20 frames per video with MedSAM2
4. Save results to `results/` directory

