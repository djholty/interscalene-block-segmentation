#!/usr/bin/env python3
"""
Entry point script for processing medical images and videos with MedSAM2.
This script calls the main script from the src directory.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run main
from process_medsam2 import main

if __name__ == "__main__":
    sys.exit(main())

