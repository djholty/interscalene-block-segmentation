"""
MedSAM2 Pipeline - Modular class-based architecture for processing medical images and videos.
"""

from .input_handlers import (
    BaseInputHandler,
    MP4InputHandler,
    DICOMInputHandler,
    TIFInputHandler,
    get_input_handler,
)
from .processor import MedSAM2Processor
from .pipeline import MedSAM2Pipeline

__all__ = [
    "BaseInputHandler",
    "MP4InputHandler",
    "DICOMInputHandler",
    "TIFInputHandler",
    "get_input_handler",
    "MedSAM2Processor",
    "MedSAM2Pipeline",
]

