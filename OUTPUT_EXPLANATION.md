# Output Explanation

## What You'll Get

When you run the pipeline, you'll get:

1. **Extracted frames** - Original video frames saved as JPEG images
2. **Visualized masks** - Frames with segmentation masks overlaid in green

## Directory Structure

```
medsam2_results/                    # Default output directory
├── MyFile_6/                       # One folder per video
│   ├── frames/                     # Original extracted frames
│   │   ├── MyFile_6_frame_000000.jpg
│   │   ├── MyFile_6_frame_000001.jpg
│   │   ├── MyFile_6_frame_000002.jpg
│   │   └── ...                     # One frame per video frame
│   │
│   └── visualizations/             # Frames with masks overlaid
│       ├── MyFile_6_frame_000000_masked.jpg
│       ├── MyFile_6_frame_000001_masked.jpg
│       ├── MyFile_6_frame_000002_masked.jpg
│       └── ...                     # One visualization per frame
│
├── MyFile_7/
│   ├── frames/
│   └── visualizations/
│
└── MyFile_8/
    ├── frames/
    └── visualizations/
```

## What the Visualizations Look Like

### Original Frame (`frames/`)
- Clean, unmodified video frame
- Standard JPEG format
- Same resolution as the video

### Masked Visualization (`visualizations/`)
- **Original frame** with a **green overlay** showing the segmentation mask
- **50% transparency** - you can see the original image through the mask
- **Green color** indicates what MedSAM2 segmented
- The mask represents what the model identified as the main object/region at the center of the frame

## Example Output Files

For a video with 100 frames, you'll get:
- **100 extracted frames** in `frames/` folder
- **100 visualizations** in `visualizations/` folder
- Total: **200 JPEG images** per video

## What the Masks Represent

The masks show:
- **What MedSAM2 segmented** when given a center point prompt
- The **largest/most prominent object** near the center of each frame
- Medical structures, lesions, or anatomical features (depending on your video content)

## File Naming Convention

- **Frames**: `{video_name}_frame_{frame_number:06d}.jpg`
  - Example: `MyFile_6_frame_000000.jpg`, `MyFile_6_frame_000001.jpg`
  
- **Visualizations**: `{video_name}_{frame_name}_masked.jpg`
  - Example: `MyFile_6_frame_000000_masked.jpg`

## Console Output

While running, you'll see:
```
Found 3 video(s) to process
Loading MedSAM2 model from MedSAM2/checkpoints/MedSAM2_latest.pt
Using config: configs/sam2.1_hiera_t512.yaml
Model loaded successfully on device: cpu

============================================================
Processing video: MyFile_6.mp4
============================================================
Extracted 150 frames from MyFile_6.mp4 (total frames: 150)
Processing 150 frames: 100%|████████████| 150/150 [00:45<00:00,  3.33it/s]
Saved 150 visualizations to medsam2_results/MyFile_6/visualizations
✓ Successfully processed MyFile_6: 150 frames visualized

============================================================
Pipeline completed!
Results saved to: medsam2_results
============================================================
```

## What to Do With the Output

1. **Browse visualizations** to see what MedSAM2 segmented
2. **Compare frames** to see how masks change across the video
3. **Use masks programmatically** - the mask data is in the visualization images
4. **Export for analysis** - use the frames and masks for further processing

## File Sizes

- **Frames**: Typically 50-500 KB each (depends on video resolution)
- **Visualizations**: Similar size to frames
- **Total**: For a 100-frame video, expect ~10-50 MB total

## Tips

- Use `--frame_interval 5` to process fewer frames and reduce output size
- Use `--max_frames 10` to limit output to first 10 frames per video
- Check the `visualizations/` folder to see the segmentation results
- The `frames/` folder contains the original frames for reference

