# Troubleshooting Guide

## Common Issues and Solutions

### 1. Hydra Config Not Found Error

**Error:**
```
hydra.errors.MissingConfigException: Cannot find primary config '.../sam2/configs/sam2.1_hiera_t512.yaml'
```

**Solution:**
The config path should be relative to the `sam2` package, not an absolute path. Use:
```bash
python process_videos_with_medsam2.py --config configs/sam2.1_hiera_t512.yaml
```

Not:
```bash
python process_videos_with_medsam2.py --config MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml
```

Hydra searches for configs in `pkg://sam2`, so the path should be relative to that.

### 2. NumPy/OpenCV Compatibility

**Error:**
```
AttributeError: _ARRAY_API not found
ImportError: numpy.core.multiarray failed to import
```

**Solution:**
```bash
source venv/bin/activate
pip install "numpy<2.0.0" opencv-python --upgrade
```

Or if MedSAM2 requires NumPy 2.x:
```bash
pip install "opencv-python>=4.11.0" --upgrade
```

### 3. Checkpoint Not Found

**Error:**
```
FileNotFoundError: Checkpoint not found: MedSAM2/checkpoints/MedSAM2_latest.pt
```

**Solution:**
```bash
cd MedSAM2
bash download.sh
cd ..
```

Or the script will attempt to auto-download it.

### 4. Hydra Build Failed During Installation

**Error:**
```
Failed to build hydra
error: failed-wheel-build-for-install
```

**Solution:**
Install Hydra separately first:
```bash
source venv/bin/activate
pip install hydra-core omegaconf
cd MedSAM2
pip install -e ".[dev]"
```

### 5. Out of Memory

**Symptoms:**
- Process killed
- CUDA out of memory errors
- System becomes unresponsive

**Solutions:**
- Process fewer frames: `--frame_interval 5 --max_frames 10`
- Use CPU instead of GPU: `--device cpu`
- Process videos one at a time
- Reduce batch size in the code if processing multiple frames

### 6. MedSAM2 Folder is Empty

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'MedSAM2/...'
ModuleNotFoundError: No module named 'sam2'
```

**Solution:**
The MedSAM2 folder needs to be populated with the MedSAM2 repository:
```bash
git clone https://github.com/bowang-lab/MedSAM2.git MedSAM2
cd MedSAM2 && pip install -e ".[dev]" && cd ..
```

### 7. Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'sam2'
```

**Solution:**
1. Make sure the MedSAM2 repository has been cloned (see issue #6 above)
2. Make sure you're in the virtual environment: `source venv/bin/activate`
3. Make sure MedSAM2 is installed: `cd MedSAM2 && pip install -e ".[dev]"`
4. Check that the script adds MedSAM2 to the path correctly

### 8. Video File Not Found

**Error:**
```
ValueError: Could not open video file: ...
```

**Solution:**
- Check that the video file exists
- Verify the file path is correct
- Check file permissions
- Ensure the video format is supported (MP4, AVI, MOV, etc.)

### 9. Device Not Available

**Error:**
```
RuntimeError: CUDA not available
```

**Solution:**
- Use CPU: `--device cpu`
- For Apple Silicon: `--device mps`
- Check PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`

## Getting Help

If you encounter other issues:

1. Check the error message carefully
2. Verify all dependencies are installed
3. Ensure you're using the virtual environment
4. Check that checkpoints are downloaded
5. Review the logs for more details

## Verification Commands

Test your installation:
```bash
source venv/bin/activate

# Test imports
python -c "from sam2.build_sam import build_sam2; print('✓ MedSAM2 OK')"

# Test config path
python -c "
import sys
sys.path.insert(0, 'MedSAM2')
from sam2.build_sam import build_sam2
model = build_sam2('configs/sam2.1_hiera_t512.yaml', 'MedSAM2/checkpoints/MedSAM2_2411.pt', device='cpu')
print('✓ Config path OK')
"
```

