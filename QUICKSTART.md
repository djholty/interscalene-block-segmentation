# Quick Start Guide: MedSAM2 Video Processing

## What This Pipeline Does

This pipeline processes MP4 video files by:
1. **Extracting frames** from MP4 videos
2. **Loading MedSAM2** with the latest checkpoint (`MedSAM2_latest.pt`)
3. **Generating masks** for each frame using center point prompts
4. **Visualizing results** by overlaying masks on original frames

## Quick Run

```bash
# Process all MP4 files in current directory
python process_videos_with_medsam2.py

# Process specific videos
python process_videos_with_medsam2.py --videos MyFile_6.mp4 MyFile_7.mp4 MyFile_8.mp4

# Process every 5th frame, max 10 frames per video
python process_videos_with_medsam2.py --frame_interval 5 --max_frames 10
```

## First Time Setup

1. **Install dependencies** (if not already installed):
```bash
pip install opencv-python numpy matplotlib tqdm
```

2. **Download MedSAM2 checkpoint** (if not present):
```bash
cd MedSAM2 && bash download.sh
```

The script will also attempt to auto-download the checkpoint if it's missing.

## Output

Results are saved to `medsam2_results/` by default:
- `frames/`: Extracted frames from videos
- `visualizations/`: Frames with masks overlaid

## Example Output Structure

```
medsam2_results/
├── MyFile_6/
│   ├── frames/
│   │   ├── MyFile_6_frame_000000.jpg
│   │   ├── MyFile_6_frame_000001.jpg
│   │   └── ...
│   └── visualizations/
│       ├── MyFile_6_frame_000000_masked.jpg
│       ├── MyFile_6_frame_000001_masked.jpg
│       └── ...
├── MyFile_7/
│   └── ...
└── MyFile_8/
    └── ...
```

## How Masks Are Generated

For each frame:
- Uses the **center point** of the image as a prompt
- MedSAM2 generates multiple mask candidates
- Selects the **best mask** (highest IoU score)
- Overlays it on the original frame with green color and 50% transparency

## Troubleshooting

### NumPy/OpenCV Compatibility
If you see NumPy version errors, try:
```bash
pip install "numpy<2" opencv-python --upgrade
```

### Checkpoint Not Found
The script will try to download automatically, or run:
```bash
cd MedSAM2 && bash download.sh
```

### Out of Memory
- Use `--frame_interval` to process fewer frames
- Use `--max_frames` to limit frames per video

## Full Documentation

See `README_VIDEO_PROCESSING.md` for complete documentation.

