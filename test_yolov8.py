#!/usr/bin/env python3
"""
Entry point script for testing YOLOv8 model.
This script calls the main test script from the src directory.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run main
from yolo.test_yolov8 import main

if __name__ == "__main__":
    sys.exit(main())

