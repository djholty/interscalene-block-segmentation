"""
Unit tests for the video processing pipeline with MedSAM2.
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

from process_videos_with_medsam2 import (
    extract_frames_from_video,
    generate_mask_with_center_point,
    visualize_mask_on_image,
)


class TestExtractFrames:
    """Tests for frame extraction functionality."""
    
    def test_extract_frames_success(self, tmp_path):
        """Test successful frame extraction from a mock video."""
        # Create a mock video file (simple test image sequence)
        video_path = tmp_path / "test_video.mp4"
        output_dir = tmp_path / "frames"
        
        # Create a simple test video using OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(video_path), fourcc, 10.0, (640, 480)
        )
        
        # Write 5 frames
        for i in range(5):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        
        # Extract frames
        frame_paths = extract_frames_from_video(
            str(video_path), str(output_dir), frame_interval=1
        )
        
        # Verify frames were extracted
        assert len(frame_paths) == 5
        assert all(os.path.exists(p) for p in frame_paths)
    
    def test_extract_frames_with_interval(self, tmp_path):
        """Test frame extraction with frame interval."""
        video_path = tmp_path / "test_video.mp4"
        output_dir = tmp_path / "frames"
        
        # Create test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(video_path), fourcc, 10.0, (640, 480)
        )
        
        for i in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        
        # Extract every 2nd frame
        frame_paths = extract_frames_from_video(
            str(video_path), str(output_dir), frame_interval=2
        )
        
        # Should extract 5 frames (every 2nd frame from 10 frames)
        assert len(frame_paths) == 5
    
    def test_extract_frames_invalid_video(self, tmp_path):
        """Test frame extraction with invalid video file."""
        video_path = tmp_path / "nonexistent.mp4"
        output_dir = tmp_path / "frames"
        
        with pytest.raises(ValueError, match="Could not open video"):
            extract_frames_from_video(
                str(video_path), str(output_dir)
            )


class TestMaskGeneration:
    """Tests for mask generation functionality."""
    
    @patch('process_videos_with_medsam2.SAM2ImagePredictor')
    def test_generate_mask_with_center_point(self, mock_predictor_class):
        """Test mask generation with center point prompt."""
        # Setup mock predictor
        mock_predictor = Mock()
        mock_predictor_class.return_value = mock_predictor
        
        # Mock predict method
        mock_masks = np.array([
            np.zeros((100, 100), dtype=bool),
            np.ones((100, 100), dtype=bool),
            np.zeros((100, 100), dtype=bool),
        ])
        mock_scores = np.array([0.5, 0.9, 0.3])
        mock_predictor.predict.return_value = (
            mock_masks, mock_scores, None
        )
        
        # Create test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Generate mask
        mask, score = generate_mask_with_center_point(
            mock_predictor, test_image
        )
        
        # Verify
        assert mask.shape == (100, 100)
        assert mask.dtype == bool
        assert score == 0.9  # Highest score
        mock_predictor.set_image.assert_called_once_with(test_image)
        mock_predictor.predict.assert_called_once()
    
    @patch('process_videos_with_medsam2.SAM2ImagePredictor')
    def test_generate_mask_with_custom_point(self, mock_predictor_class):
        """Test mask generation with custom point coordinates."""
        mock_predictor = Mock()
        mock_predictor_class.return_value = mock_predictor
        
        mock_masks = np.array([np.ones((100, 100), dtype=bool)])
        mock_scores = np.array([0.8])
        mock_predictor.predict.return_value = (
            mock_masks, mock_scores, None
        )
        
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        custom_point = np.array([[50, 50]], dtype=np.float32)
        
        mask, score = generate_mask_with_center_point(
            mock_predictor, test_image, point_coords=custom_point
        )
        
        assert mask.shape == (100, 100)
        assert score == 0.8


class TestVisualization:
    """Tests for visualization functionality."""
    
    def test_visualize_mask_on_image(self):
        """Test mask visualization overlay."""
        # Create test image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:] = [128, 128, 128]  # Gray image
        
        # Create test mask (center region)
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True
        
        # Visualize
        result = visualize_mask_on_image(
            image, mask, alpha=0.5, color=(0, 255, 0)
        )
        
        # Verify
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8
        # Masked region should be different from original
        assert not np.array_equal(result[40:60, 40:60], image[40:60, 40:60])
    
    def test_visualize_mask_empty_mask(self):
        """Test visualization with empty mask."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=bool)
        
        result = visualize_mask_on_image(image, mask)
        
        # Should return original image (no mask to overlay)
        assert np.array_equal(result, image)
    
    def test_visualize_mask_uint8_mask(self):
        """Test visualization with uint8 mask instead of bool."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255
        
        result = visualize_mask_on_image(image, mask)
        
        # Should work with uint8 mask
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_visualize_mask_different_sizes(self):
        """Test that visualization handles size mismatches gracefully."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=bool)  # Different size
        
        # Should raise an error or handle gracefully
        with pytest.raises((ValueError, IndexError)):
            visualize_mask_on_image(image, mask)
    
    @patch('process_videos_with_medsam2.build_sam2')
    def test_load_model_missing_checkpoint(self, mock_build_sam2):
        """Test model loading with missing checkpoint."""
        from process_videos_with_medsam2 import load_medsam2_model
        
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            load_medsam2_model(
                checkpoint_path="/nonexistent/path/checkpoint.pt",
                config_path="config.yaml",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

