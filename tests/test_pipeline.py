"""
Unit tests for the modular MedSAM2 pipeline.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from medsam2_pipeline.input_handlers import (
    BaseInputHandler,
    MP4InputHandler,
    DICOMInputHandler,
    TIFInputHandler,
    get_input_handler,
)
from medsam2_pipeline.processor import MedSAM2Processor
from medsam2_pipeline.pipeline import MedSAM2Pipeline


class TestInputHandlers:
    """Tests for input handler classes."""
    
    def test_mp4_handler_supports(self):
        """Test MP4 handler file support detection."""
        assert MP4InputHandler.supports("test.mp4")
        assert MP4InputHandler.supports("test.AVI")
        assert MP4InputHandler.supports("test.mov")
        assert not MP4InputHandler.supports("test.jpg")
        assert not MP4InputHandler.supports("test.dcm")
    
    def test_mp4_handler_extract(self, tmp_path):
        """Test MP4 handler frame extraction."""
        # Create a mock video file
        video_path = tmp_path / "test_video.mp4"
        output_dir = tmp_path / "frames"
        
        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (640, 480))
        
        for i in range(5):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        
        # Extract frames
        handler = MP4InputHandler()
        images = handler.extract_images(str(video_path), str(output_dir))
        
        assert len(images) == 5
        assert all(os.path.exists(path) for path, _ in images)
        assert all(img.shape[2] == 3 for _, img in images)  # RGB format
    
    def test_get_input_handler_mp4(self):
        """Test getting handler for MP4 file."""
        handler = get_input_handler("test.mp4")
        assert isinstance(handler, MP4InputHandler)
    
    def test_get_input_handler_unsupported(self):
        """Test getting handler for unsupported file."""
        with pytest.raises(ValueError, match="No handler found"):
            get_input_handler("test.xyz")
    
    @pytest.mark.skipif(
        not pytest.importorskip("pydicom", reason="pydicom not installed"),
        reason="pydicom not available"
    )
    def test_dicom_handler_supports(self):
        """Test DICOM handler file support detection."""
        # This would require actual DICOM files to test properly
        # Just test the method exists
        assert hasattr(DICOMInputHandler, 'supports')
    
    @pytest.mark.skipif(
        not pytest.importorskip("tifffile", reason="tifffile not installed"),
        reason="tifffile not available"
    )
    def test_tif_handler_supports(self):
        """Test TIF handler file support detection."""
        assert TIFInputHandler.supports("test.tif")
        assert TIFInputHandler.supports("test.TIFF")
        assert not TIFInputHandler.supports("test.jpg")


class TestMedSAM2Processor:
    """Tests for MedSAM2Processor class."""
    
    @patch('medsam2_pipeline.processor.build_sam2')
    @patch('medsam2_pipeline.processor.SAM2ImagePredictor')
    def test_processor_initialization(self, mock_predictor_class, mock_build_sam2):
        """Test processor initialization."""
        # Setup mocks
        mock_model = Mock()
        mock_build_sam2.return_value = mock_model
        mock_predictor = Mock()
        mock_predictor_class.return_value = mock_predictor
        
        # Create processor
        processor = MedSAM2Processor(
            checkpoint_path="test.pt",
            config_path="config.yaml",
            device="cpu"
        )
        
        assert processor.is_loaded
        mock_build_sam2.assert_called_once()
    
    @patch('medsam2_pipeline.processor.build_sam2')
    @patch('medsam2_pipeline.processor.SAM2ImagePredictor')
    def test_generate_mask(self, mock_predictor_class, mock_build_sam2):
        """Test mask generation."""
        # Setup mocks
        mock_model = Mock()
        mock_build_sam2.return_value = mock_model
        mock_predictor = Mock()
        mock_predictor_class.return_value = mock_predictor
        
        # Mock predict method
        mock_masks = np.array([
            np.zeros((100, 100), dtype=bool),
            np.ones((100, 100), dtype=bool),
        ])
        mock_scores = np.array([0.5, 0.9])
        mock_predictor.predict.return_value = (mock_masks, mock_scores, None)
        
        # Create processor and generate mask
        processor = MedSAM2Processor("test.pt", "config.yaml")
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        mask, score = processor.generate_mask(test_image)
        
        assert mask.shape == (100, 100)
        assert mask.dtype == bool
        assert score == 0.9  # Highest score


class TestMedSAM2Pipeline:
    """Tests for MedSAM2Pipeline class."""
    
    @patch('medsam2_pipeline.pipeline.MedSAM2Processor')
    def test_pipeline_initialization(self, mock_processor_class):
        """Test pipeline initialization."""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        
        pipeline = MedSAM2Pipeline("test.pt", "config.yaml")
        
        assert pipeline.processor is not None
        mock_processor_class.assert_called_once()
    
    @patch('medsam2_pipeline.pipeline.MedSAM2Processor')
    @patch('medsam2_pipeline.pipeline.get_input_handler')
    def test_process_file(self, mock_get_handler, mock_processor_class):
        """Test processing a single file."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor.generate_mask.return_value = (
            np.ones((100, 100), dtype=bool), 0.9
        )
        mock_processor_class.return_value = mock_processor
        
        mock_handler = Mock()
        mock_handler.extract_images.return_value = [
            ("frame1.jpg", np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)),
            ("frame2.jpg", np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)),
        ]
        mock_get_handler.return_value = mock_handler
        
        # Create pipeline and process
        pipeline = MedSAM2Pipeline("test.pt", "config.yaml")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = pipeline.process_file("test.mp4", tmpdir)
            
            assert result['num_images'] == 2
            assert 'visualizations_dir' in result
            assert len(result['visualization_paths']) == 2


class TestIntegration:
    """Integration tests (may require actual files)."""
    
    def test_handler_normalize_image(self):
        """Test image normalization in base handler."""
        handler = MP4InputHandler()
        
        # Test grayscale
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        rgb = handler._normalize_image(gray)
        assert rgb.shape == (100, 100, 3)
        
        # Test BGR
        bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        rgb = handler._normalize_image(bgr)
        assert rgb.shape == (100, 100, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

