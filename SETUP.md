# Setup Instructions

## Create and Activate Virtual Environment

### On macOS/Linux:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### On Windows:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

## Install Dependencies

### 1. Install Basic Requirements
```bash
# Make sure venv is activated (you should see (venv) in your prompt)
pip install -r requirements.txt
```

### 2. Install MedSAM2 Dependencies
```bash
cd MedSAM2
pip install -e ".[dev]"
cd ..
```

### 3. Download MedSAM2 Checkpoints
```bash
cd MedSAM2
bash download.sh
cd ..
```

## Verify Installation

Test that everything is installed correctly:
```bash
python -c "import cv2, numpy, torch; print('✓ All imports successful')"
```

## Deactivate Virtual Environment

When you're done working:
```bash
deactivate
```

## Troubleshooting

### NumPy/OpenCV Compatibility Issues
If you encounter NumPy 2.x compatibility issues:
```bash
pip install "numpy<2.0.0" --upgrade
pip install opencv-python --upgrade
```

### PyTorch Installation
For CUDA support, install PyTorch from the official site:
```bash
# Visit https://pytorch.org/get-started/locally/ for the correct command
# Example for CUDA 12.4:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### MedSAM2 Installation Issues
If MedSAM2 installation fails:
1. Make sure you're in the MedSAM2 directory
2. Check that all system dependencies are installed (see MedSAM2/README.md)
3. Try installing without dev dependencies: `pip install -e .`

