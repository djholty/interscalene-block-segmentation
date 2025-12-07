# Interscalene Block Segmentation

A modular pipeline for processing medical images and videos with MedSAM2 for interscalene block segmentation.

## Features

- **Multiple Input Formats**: MP4/AVI/MOV videos, DICOM medical images, TIF/TIFF images
- **MedSAM2 Integration**: State-of-the-art medical image segmentation
- **Modular Architecture**: Clean, extensible class-based design
- **Batch Processing**: Process single files or entire directories

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/interscalene-block-segmentation.git
cd interscalene-block-segmentation
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
cd MedSAM2 && pip install -e ".[dev]" && cd ..
```

4. Download MedSAM2 checkpoints:
```bash
cd MedSAM2 && bash download.sh && cd ..
```

### Usage

```bash
# Process MP4 video
python process_medsam2.py -i video.mp4 -o results

# Process DICOM directory
python process_medsam2.py -i dicom_folder/ -o results

# Process TIF file
python process_medsam2.py -i image.tif -o results
```

## Architecture

The pipeline consists of modular components:

- **Input Handlers**: Support for MP4, DICOM, and TIF formats
- **MedSAM2 Processor**: Model loading and mask generation
- **Pipeline**: Main orchestration class

See [README_MODULAR.md](README_MODULAR.md) for detailed documentation.

## Project Structure

```
.
├── medsam2_pipeline/          # Modular pipeline package
│   ├── input_handlers.py      # Input format handlers
│   ├── processor.py            # MedSAM2 processor
│   └── pipeline.py             # Main pipeline
├── MedSAM2/                   # MedSAM2 model code
├── process_medsam2.py          # Main script (modular)
├── process_videos_with_medsam2.py  # Legacy script
├── tests/                      # Unit tests
└── requirements.txt            # Dependencies
```

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- pydicom, SimpleITK (for DICOM)
- tifffile (for TIF)

See `requirements.txt` for complete list.

## Documentation

- [README_MODULAR.md](README_MODULAR.md) - Detailed modular pipeline documentation
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions

## License

[Add your license here]

## Citation

If you use this code, please cite:
- MedSAM2 paper
- SAM2 paper
- This repository

## Contributing

[Add contribution guidelines]

