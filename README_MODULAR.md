# Modular MedSAM2 Pipeline

A class-based, modular pipeline for processing medical images and videos with MedSAM2.

## Features

- **Multiple Input Formats**: MP4/AVI/MOV videos, DICOM medical images, TIF/TIFF images
- **Modular Architecture**: Clean separation of concerns with classes
- **Extensible**: Easy to add new input formats or processing steps
- **Batch Processing**: Process single files or entire directories

## Architecture

### Classes

1. **Input Handlers** (`medsam2_pipeline/input_handlers.py`)
   - `BaseInputHandler`: Abstract base class
   - `MP4InputHandler`: Handles video files
   - `DICOMInputHandler`: Handles DICOM medical images
   - `TIFInputHandler`: Handles TIF/TIFF images (including multi-page)

2. **Processor** (`medsam2_pipeline/processor.py`)
   - `MedSAM2Processor`: Manages MedSAM2 model loading and mask generation

3. **Pipeline** (`medsam2_pipeline/pipeline.py`)
   - `MedSAM2Pipeline`: Main orchestration class

## Installation

```bash
# Activate virtual environment
source venv/bin/activate

# Install additional dependencies
pip install pydicom SimpleITK tifffile

# Or install all requirements
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
# Process MP4 video
python process_medsam2.py --input video.mp4 --output results

# Process DICOM directory
python process_medsam2.py --input dicom_folder/ --output results

# Process TIF file
python process_medsam2.py --input image.tif --output results

# Process all supported files in directory
python process_medsam2.py --input input_dir/ --output results

# With options
python process_medsam2.py \
    --input video.mp4 \
    --output results \
    --frame_interval 5 \
    --max_images 10 \
    --device cuda
```

### Python API

```python
from medsam2_pipeline import MedSAM2Pipeline

# Initialize pipeline
pipeline = MedSAM2Pipeline(
    checkpoint_path="MedSAM2/checkpoints/MedSAM2_latest.pt",
    config_path="configs/sam2.1_hiera_t512.yaml",
    device="cuda"  # or "mps", "cpu", or None for auto-detect
)

# Process a single file
result = pipeline.process_file(
    file_path="video.mp4",
    output_dir="results",
    frame_interval=1,
    max_images=None,
    alpha=0.5,
    mask_color=(0, 255, 0)
)

# Process a directory
results = pipeline.process_directory(
    input_dir="input_folder/",
    output_dir="results",
    frame_interval=1,
    max_images=None
)
```

## Supported Formats

### MP4/AVI/MOV Videos
- Extracts frames from video files
- Supports: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`

### DICOM Medical Images
- Single DICOM files (`.dcm`, `.dicom`)
- DICOM series (directories)
- Automatic windowing and normalization
- Requires: `pydicom`, `SimpleITK`

### TIF/TIFF Images
- Single and multi-page TIF files
- Multi-channel support
- Requires: `tifffile`

## Output Structure

```
output_dir/
├── file_name/
│   ├── frames/              # Extracted images
│   │   ├── file_name_frame_000000.jpg
│   │   └── ...
│   └── visualizations/      # Mask visualizations
│       ├── file_name_frame_000000_masked.jpg
│       └── ...
```

## Extending the Pipeline

### Adding a New Input Format

1. Create a new handler class inheriting from `BaseInputHandler`:

```python
class MyFormatHandler(BaseInputHandler):
    @classmethod
    def supports(cls, file_path: str) -> bool:
        return file_path.endswith('.myformat')
    
    def extract_images(self, file_path, output_dir, ...):
        # Implementation
        pass
    
    def get_metadata(self, file_path):
        # Implementation
        pass
```

2. Register it in `get_input_handler()` function

### Custom Processing

```python
from medsam2_pipeline import MedSAM2Processor

processor = MedSAM2Processor("checkpoint.pt", "config.yaml")

# Generate mask for custom image
mask, score = processor.generate_mask(
    image=my_image_array,
    point_coords=[[x, y]],  # Optional custom point
    multimask_output=True
)
```

## Command Line Options

- `--input, -i`: Input file or directory (required)
- `--output, -o`: Output directory (default: `medsam2_results`)
- `--checkpoint`: Path to MedSAM2 checkpoint
- `--config`: Path to MedSAM2 config (relative to sam2 package)
- `--frame_interval`: Process every Nth image (default: 1)
- `--max_images`: Maximum images per file (default: all)
- `--device`: Device to use (`cuda`, `mps`, `cpu`, or auto-detect)
- `--extensions`: Specific file extensions to process
- `--alpha`: Mask overlay transparency (0.0-1.0, default: 0.5)
- `--mask_color`: Mask color in RGB (default: 0 255 0)

## Examples

### Process Medical Video
```bash
python process_medsam2.py \
    --input ultrasound_video.mp4 \
    --output results/ultrasound \
    --frame_interval 2
```

### Process DICOM Series
```bash
python process_medsam2.py \
    --input CT_scan_folder/ \
    --output results/ct_scan \
    --extensions .dcm
```

### Process Multi-page TIF
```bash
python process_medsam2.py \
    --input microscopy_stack.tif \
    --output results/microscopy \
    --max_images 50
```

## Testing

```bash
# Run unit tests
pytest tests/test_pipeline.py -v

# Run with coverage
pytest tests/test_pipeline.py --cov=medsam2_pipeline
```

## Migration from Old Script

The old `process_videos_with_medsam2.py` script is still available for backward compatibility. The new modular pipeline (`process_medsam2.py`) provides:

- Better code organization
- Support for more formats (DICOM, TIF)
- Easier extensibility
- Cleaner API

## Troubleshooting

### DICOM Support
If you get import errors for DICOM:
```bash
pip install pydicom SimpleITK
```

### TIF Support
If you get import errors for TIF:
```bash
pip install tifffile
```

### Memory Issues
- Use `--frame_interval` to process fewer images
- Use `--max_images` to limit per-file processing
- Process files individually instead of directories

