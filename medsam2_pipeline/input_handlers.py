"""
Input handlers for different file formats (MP4, DICOM, TIF).
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Optional
import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class BaseInputHandler(ABC):
    """
    Base class for input handlers that extract images from different file formats.
    
    All input handlers must implement methods to:
    1. Check if a file is supported
    2. Extract images/frames from the file
    3. Get metadata about the file
    """
    
    @classmethod
    @abstractmethod
    def supports(cls, file_path: str) -> bool:
        """
        Check if this handler supports the given file.
        
        Args:
            file_path (str): Path to the file.
            
        Returns:
            bool: True if this handler can process the file.
        """
        pass
    
    @abstractmethod
    def extract_images(
        self,
        file_path: str,
        output_dir: str,
        frame_interval: int = 1,
        max_images: Optional[int] = None,
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Extract images from the input file.
        
        Args:
            file_path (str): Path to the input file.
            output_dir (str): Directory to save extracted images.
            frame_interval (int): Extract every Nth image (default: 1).
            max_images (int, optional): Maximum number of images to extract.
            
        Returns:
            List[Tuple[str, np.ndarray]]: List of (image_path, image_array) tuples.
                                         image_array is in RGB format (H, W, 3), uint8.
        """
        pass
    
    @abstractmethod
    def get_metadata(self, file_path: str) -> dict:
        """
        Get metadata about the input file.
        
        Args:
            file_path (str): Path to the file.
            
        Returns:
            dict: Metadata dictionary with keys like 'num_images', 'shape', etc.
        """
        pass
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to RGB uint8 format.
        
        Args:
            image (np.ndarray): Input image in various formats.
            
        Returns:
            np.ndarray: Image in RGB format (H, W, 3), uint8.
        """
        # Handle grayscale
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        # Handle single channel
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        # Handle BGR to RGB
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # Check if it's BGR (OpenCV format)
            if image.dtype == np.uint8:
                # Assume BGR if from OpenCV, convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure uint8
        if image.dtype != np.uint8:
            # Normalize to 0-255 range
            if image.max() > 1.0:
                image = np.clip(image, 0, 255).astype(np.uint8)
            else:
                image = (image * 255).astype(np.uint8)
        
        return image


class MP4InputHandler(BaseInputHandler):
    """Handler for MP4 video files."""
    
    @classmethod
    def supports(cls, file_path: str) -> bool:
        """Check if file is an MP4 video."""
        return file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
    
    def extract_images(
        self,
        file_path: str,
        output_dir: str,
        frame_interval: int = 1,
        max_images: Optional[int] = None,
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Extract frames from MP4 video.
        
        Args:
            file_path (str): Path to MP4 file.
            output_dir (str): Directory to save frames.
            frame_interval (int): Extract every Nth frame.
            max_images (int, optional): Maximum frames to extract.
            
        Returns:
            List[Tuple[str, np.ndarray]]: List of (path, image_array) tuples.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {file_path}")
        
        video_name = Path(file_path).stem
        images = []
        frame_count = 0
        saved_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    if max_images and saved_count >= max_images:
                        break
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Save frame
                    frame_filename = f"{video_name}_frame_{saved_count:06d}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    
                    images.append((frame_path, frame_rgb))
                    saved_count += 1
                
                frame_count += 1
        finally:
            cap.release()
        
        logger.info(
            f"Extracted {len(images)} frames from {file_path} "
            f"(total frames: {frame_count})"
        )
        return images
    
    def get_metadata(self, file_path: str) -> dict:
        """Get video metadata."""
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {file_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        
        return {
            'num_images': total_frames,
            'fps': fps,
            'shape': (height, width, 3),
            'format': 'video',
        }


class DICOMInputHandler(BaseInputHandler):
    """Handler for DICOM medical image files."""
    
    @classmethod
    def supports(cls, file_path: str) -> bool:
        """Check if file is a DICOM file."""
        try:
            import pydicom
            return pydicom.misc.is_dicom(file_path)
        except ImportError:
            logger.warning("pydicom not installed. DICOM support unavailable.")
            return False
        except Exception:
            return False
    
    def extract_images(
        self,
        file_path: str,
        output_dir: str,
        frame_interval: int = 1,
        max_images: Optional[int] = None,
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Extract images from DICOM file(s).
        
        Args:
            file_path (str): Path to DICOM file or directory.
            output_dir (str): Directory to save images.
            frame_interval (int): Extract every Nth slice (for multi-slice DICOM).
            max_images (int, optional): Maximum images to extract.
            
        Returns:
            List[Tuple[str, np.ndarray]]: List of (path, image_array) tuples.
        """
        try:
            import pydicom
            import SimpleITK as sitk
        except ImportError as e:
            raise ImportError(
                f"DICOM support requires pydicom and SimpleITK. Install with: "
                f"pip install pydicom SimpleITK"
            ) from e
        
        os.makedirs(output_dir, exist_ok=True)
        file_path_obj = Path(file_path)
        images = []
        
        # Handle single DICOM file
        if file_path_obj.is_file():
            dicom_files = [file_path]
        # Handle directory of DICOM files
        elif file_path_obj.is_dir():
            dicom_files = sorted([
                str(f) for f in file_path_obj.rglob('*')
                if pydicom.misc.is_dicom(str(f))
            ])
        else:
            raise ValueError(f"DICOM path not found: {file_path}")
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in: {file_path}")
        
        base_name = file_path_obj.stem if file_path_obj.is_file() else file_path_obj.name
        
        # Try to read as 3D volume using SimpleITK
        try:
            reader = sitk.ImageSeriesReader()
            if file_path_obj.is_dir():
                dicom_names = reader.GetGDCMSeriesFileNames(str(file_path))
                if dicom_names:
                    reader.SetFileNames(dicom_names)
                    image_3d = reader.Execute()
                    image_array = sitk.GetArrayFromImage(image_3d)
                    
                    # Normalize to 0-255
                    if image_array.max() > 255:
                        image_array = np.clip(image_array, image_array.min(), image_array.max())
                        image_array = ((image_array - image_array.min()) / 
                                     (image_array.max() - image_array.min()) * 255).astype(np.uint8)
                    else:
                        image_array = image_array.astype(np.uint8)
                    
                    # Extract slices
                    saved_count = 0
                    for slice_idx in range(0, len(image_array), frame_interval):
                        if max_images and saved_count >= max_images:
                            break
                        
                        slice_data = image_array[slice_idx]
                        slice_rgb = self._normalize_image(slice_data)
                        
                        slice_filename = f"{base_name}_slice_{saved_count:06d}.jpg"
                        slice_path = os.path.join(output_dir, slice_filename)
                        Image.fromarray(slice_rgb).save(slice_path)
                        
                        images.append((slice_path, slice_rgb))
                        saved_count += 1
                    
                    logger.info(
                        f"Extracted {len(images)} slices from DICOM series "
                        f"(total slices: {len(image_array)})"
                    )
                    return images
        except Exception as e:
            logger.warning(f"Could not read as DICOM series, trying individual files: {e}")
        
        # Fallback: read individual DICOM files
        saved_count = 0
        for idx, dicom_file in enumerate(dicom_files):
            if idx % frame_interval != 0:
                continue
            if max_images and saved_count >= max_images:
                break
            
            try:
                ds = pydicom.dcmread(dicom_file)
                pixel_array = ds.pixel_array
                
                # Apply windowing if available
                if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                    center = float(ds.WindowCenter)
                    width = float(ds.WindowWidth)
                    pixel_array = np.clip(
                        pixel_array,
                        center - width / 2,
                        center + width / 2
                    )
                
                # Normalize to 0-255
                if pixel_array.max() > 255:
                    pixel_array = ((pixel_array - pixel_array.min()) / 
                                 (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
                else:
                    pixel_array = pixel_array.astype(np.uint8)
                
                image_rgb = self._normalize_image(pixel_array)
                
                image_filename = f"{base_name}_dicom_{saved_count:06d}.jpg"
                image_path = os.path.join(output_dir, image_filename)
                Image.fromarray(image_rgb).save(image_path)
                
                images.append((image_path, image_rgb))
                saved_count += 1
            except Exception as e:
                logger.warning(f"Failed to read DICOM file {dicom_file}: {e}")
                continue
        
        logger.info(f"Extracted {len(images)} images from {len(dicom_files)} DICOM files")
        return images
    
    def get_metadata(self, file_path: str) -> dict:
        """Get DICOM metadata."""
        try:
            import pydicom
            import SimpleITK as sitk
        except ImportError:
            raise ImportError("DICOM support requires pydicom and SimpleITK")
        
        file_path_obj = Path(file_path)
        
        # Try to get series info
        try:
            reader = sitk.ImageSeriesReader()
            if file_path_obj.is_dir():
                dicom_names = reader.GetGDCMSeriesFileNames(str(file_path))
                if dicom_names:
                    reader.SetFileNames(dicom_names)
                    image_3d = reader.Execute()
                    image_array = sitk.GetArrayFromImage(image_3d)
                    return {
                        'num_images': len(image_array),
                        'shape': image_array.shape,
                        'format': 'dicom_series',
                    }
        except Exception:
            pass
        
        # Fallback: count individual files
        if file_path_obj.is_dir():
            dicom_files = [f for f in file_path_obj.rglob('*') if pydicom.misc.is_dicom(str(f))]
            if dicom_files:
                ds = pydicom.dcmread(dicom_files[0])
                return {
                    'num_images': len(dicom_files),
                    'shape': ds.pixel_array.shape,
                    'format': 'dicom_files',
                }
        
        # Single file
        ds = pydicom.dcmread(file_path)
        return {
            'num_images': 1,
            'shape': ds.pixel_array.shape,
            'format': 'dicom_single',
        }


class TIFInputHandler(BaseInputHandler):
    """Handler for TIF/TIFF image files (including multi-page and multi-channel)."""
    
    @classmethod
    def supports(cls, file_path: str) -> bool:
        """Check if file is a TIF/TIFF image."""
        return file_path.lower().endswith(('.tif', '.tiff'))
    
    def extract_images(
        self,
        file_path: str,
        output_dir: str,
        frame_interval: int = 1,
        max_images: Optional[int] = None,
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Extract images from TIF file (supports multi-page and multi-channel).
        
        Args:
            file_path (str): Path to TIF file.
            output_dir (str): Directory to save images.
            frame_interval (int): Extract every Nth page/channel.
            max_images (int, optional): Maximum images to extract.
            
        Returns:
            List[Tuple[str, np.ndarray]]: List of (path, image_array) tuples.
        """
        try:
            import tifffile
        except ImportError:
            raise ImportError(
                "TIF support requires tifffile. Install with: pip install tifffile"
            )
        
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(file_path).stem
        images = []
        
        # Read TIF file
        with tifffile.TiffFile(file_path) as tif:
            # Get number of pages
            num_pages = len(tif.pages)
            
            saved_count = 0
            for page_idx in range(0, num_pages, frame_interval):
                if max_images and saved_count >= max_images:
                    break
                
                # Read page
                page_data = tif.pages[page_idx].asarray()
                
                # Handle different data types and shapes
                image_rgb = self._normalize_image(page_data)
                
                # Save image
                image_filename = f"{base_name}_page_{saved_count:06d}.jpg"
                image_path = os.path.join(output_dir, image_filename)
                Image.fromarray(image_rgb).save(image_path)
                
                images.append((image_path, image_rgb))
                saved_count += 1
        
        logger.info(
            f"Extracted {len(images)} pages from TIF file "
            f"(total pages: {num_pages})"
        )
        return images
    
    def get_metadata(self, file_path: str) -> dict:
        """Get TIF metadata."""
        try:
            import tifffile
        except ImportError:
            raise ImportError("TIF support requires tifffile")
        
        with tifffile.TiffFile(file_path) as tif:
            num_pages = len(tif.pages)
            first_page = tif.pages[0].asarray()
            
            return {
                'num_images': num_pages,
                'shape': first_page.shape,
                'format': 'tif',
            }


def get_input_handler(file_path: str) -> BaseInputHandler:
    """
    Get the appropriate input handler for a file.
    
    Args:
        file_path (str): Path to the input file.
        
    Returns:
        BaseInputHandler: Handler instance for the file type.
        
    Raises:
        ValueError: If no handler supports the file.
    """
    handlers = [
        MP4InputHandler,
        TIFInputHandler,
        DICOMInputHandler,  # Check DICOM last as it may require file reading
    ]
    
    for handler_class in handlers:
        try:
            if handler_class.supports(file_path):
                return handler_class()
        except Exception as e:
            # Skip handlers that fail (e.g., missing dependencies)
            logger.debug(f"Handler {handler_class.__name__} failed: {e}")
            continue
    
    raise ValueError(
        f"No handler found for file: {file_path}. "
        f"Supported formats: MP4/AVI/MOV, DICOM, TIF/TIFF"
    )

