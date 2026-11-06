"""
Color Mapper Processor Module.

This module provides the computational core for color mapping and palette-based
color quantification. It maps masked image pixels to their nearest neighbor
in a defined RGB palette using CIELAB Euclidean color distance (Delta E 1976),
and calculates color distributions and region-of-interest (ROI) statistics.
"""

from __future__ import annotations
import numpy as np
import cv2
import json
from pathlib import Path
from loguru import logger


class ColorMapperProcessor:
    """
    Handles image processing, color space conversions, and palette management
    for the Color Mapper module.
    """

    config_path: Path
    color_map_RGB: dict[str, tuple[int, int, int]]
    palette_keys: list[str]
    palette_rgb: np.ndarray | None
    palette_bgr: np.ndarray | None
    palette_lab: np.ndarray | None
    _palette_lab_T: np.ndarray | None
    _palette_half_norm_sq: np.ndarray | None
    frame: np.ndarray | None
    mask: np.ndarray | None
    _aligned_mask: np.ndarray | None
    _cached_mapped_frame: np.ndarray | None
    _cached_preview_frame: np.ndarray | None
    _cached_index_map: np.ndarray | None
    _cached_distribution: dict[str, dict]
    _needs_remapping: bool
    last_distribution: dict[str, dict]

    def __init__(self, config_path: str | Path) -> None:
        """
        Initialize the ColorMapperProcessor with a palette configuration path.
        """
        self.config_path = Path(config_path)
        
        self.color_map_RGB = {}
        self.palette_keys = []
        self.palette_rgb = None
        self.palette_bgr = None
        self.palette_lab = None
        self._palette_lab_T = None
        self._palette_half_norm_sq = None
        
        self.frame = None
        self.mask = None
        self._aligned_mask = None
        
        # Caching
        self._cached_mapped_frame = None
        self._cached_preview_frame = None
        self._cached_index_map = None
        self._cached_distribution = {}
        self._needs_remapping = True
        self.last_distribution = {}
        
        self.load_palette()
        self.precompute_palette()

    def load_palette(self) -> None:
        """
        Load the color palette from the configured JSON file.

        If the file does not exist or fails to load, a default palette
        is loaded and saved back to the file.
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    data = json.load(f)
                    self.color_map_RGB = {k: tuple(v) for k, v in data.items()}
                return
            except Exception as e:
                logger.error(f"ColorMapperProcessor: Failed to load palette: {e}")
                
        # Default fallback
        self.color_map_RGB = {
            'Black': (0,0,0),
            'White': (255,255,255),
            'B1': (247, 248, 232),
            'B2': (243, 244, 192),
            'B3': (234, 235, 137),
            'B4': (200, 206, 57),
            'E5': (155, 124, 45),
            'E6': (111, 85, 34)
        }
        self.save_palette()

    def save_palette(self) -> None:
        """
        Persist the current palette dictionary (color_map_RGB) to the JSON file.
        """
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.color_map_RGB, f, indent=4)
        except Exception as e:
            logger.error(f"ColorMapperProcessor: Failed to save palette: {e}")

    def precompute_palette(self) -> None:
        """
        Precompute numpy arrays for RGB and CIELAB color representations.

        Transforms palette RGB colors into a normalized float32 CIELAB array
        to accelerate vector distance calculations during mapping.
        """
        self.palette_keys = list(self.color_map_RGB.keys())
        if not self.palette_keys:
            self.palette_rgb = None
            self.palette_bgr = None
            self.palette_lab = None
            self._palette_lab_T = None
            self._palette_half_norm_sq = None
            self._cached_index_map = None
            return
            
        self.palette_rgb = np.array([self.color_map_RGB[k] for k in self.palette_keys], dtype=np.uint8)
        self.palette_bgr = self.palette_rgb[:, ::-1].copy()
        self.palette_lab = cv2.cvtColor(self.palette_rgb[None, :, :], cv2.COLOR_RGB2Lab)[0].astype(np.float32)
        self._palette_lab_T = np.ascontiguousarray(self.palette_lab.T)
        self._palette_half_norm_sq = 0.5 * np.sum(self.palette_lab * self.palette_lab, axis=1)
        self._cached_index_map = None
        self._needs_remapping = True

    def add_key(self, new_name: str) -> bool:
        """
        Add a new color key initialized to black (0, 0, 0) to the palette.
        """
        new_name = new_name.strip()
        if new_name and new_name not in self.color_map_RGB:
            self.color_map_RGB[new_name] = (0, 0, 0)
            self.precompute_palette()
            return True
        return False

    def rename_key(self, old_name: str, new_name: str) -> bool:
        """
        Rename an existing color key in the palette while preserving its RGB value and order.
        """
        new_name = new_name.strip()
        if not new_name or new_name == old_name or new_name in self.color_map_RGB:
            return False
            
        new_palette = {}
        for k, v in self.color_map_RGB.items():
            if k == old_name:
                new_palette[new_name] = v
            else:
                new_palette[k] = v
        self.color_map_RGB = new_palette
        self.precompute_palette()
        return True

    def delete_key(self, name: str) -> bool:
        """
        Remove a color key from the palette.
        """
        if name in self.color_map_RGB:
            del self.color_map_RGB[name]
            self.precompute_palette()
            return True
        return False

    def reset_inputs(self) -> None:
        """
        Reset and clear input frames, masks, and cached mapping results.
        """
        self.frame = None
        self.mask = None
        self._aligned_mask = None
        self._cached_mapped_frame = None
        self._cached_preview_frame = None
        self._cached_index_map = None
        self._needs_remapping = True

    def get_aligned_mask(self) -> np.ndarray | None:
        """
        Return the active mask aligned to the current frame's spatial dimensions, caching the resized result.
        """
        if self.mask is None or self.frame is None:
            return self.mask
        fh, fw = self.frame.shape[:2]
        if self._aligned_mask is not None and self._aligned_mask.shape[:2] == (fh, fw):
            return self._aligned_mask
        if self.mask.shape[:2] == (fh, fw):
            self._aligned_mask = self.mask
        else:
            self._aligned_mask = cv2.resize(self.mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
        return self._aligned_mask

    def ingest_data(self, data: np.ndarray | tuple[np.ndarray, np.ndarray], is_pair: bool = False) -> bool:
        """
        Ingest raw input data and separate image frame from segmentation mask.
        """
        if is_pair:
            frame = data[0]
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            self.frame = frame
            self.mask = data[1]
            self._aligned_mask = None
            self._cached_index_map = None
            self._needs_remapping = True # A pair triggers automatic update
            return True
            
        if not isinstance(data, np.ndarray):
            return False
            
        is_mask = False
        if len(data.shape) == 2:
            is_mask = True
        elif len(data.shape) == 3:
            if data.shape[2] == 1:
                is_mask = True
            elif data.size > 0 and data[0, 0, 0] == data[0, 0, 1] == data[0, 0, 2]:
                if np.array_equal(data[:, :, 0], data[:, :, 1]) and np.array_equal(data[:, :, 1], data[:, :, 2]):
                    is_mask = True
        
        if is_mask:
            self.mask = data[:,:,0] if len(data.shape) == 3 else data
            self._aligned_mask = None
            self._cached_index_map = None
            self._needs_remapping = True # New mask triggers update
        else:
            frame = data
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            self.frame = frame
            self._aligned_mask = None
            # New frame arrives: we MUST clear the previous results
            # so the user sees the live original frame instead of a static cached map.
            self._cached_mapped_frame = None
            self._cached_preview_frame = None
            self._cached_index_map = None
            # NO automatic needs_remapping on single frame
            
        return True

    def set_full_mask(self) -> bool:
        """
        Create a full-white mask covering the entire frame dimensions.
        """
        if self.frame is not None:
            self.mask = np.full(self.frame.shape[:2], 255, dtype=np.uint8)
            self._aligned_mask = self.mask
            self._cached_index_map = None
            self._needs_remapping = True
            return True
        return False

    def process_map(self, show_original: bool, is_comparing: bool) -> tuple[np.ndarray | None, dict[str, dict], str]:
        """
        Process color mapping on the frame using the active mask and palette.

        Calculates the nearest palette color in CIELAB color space for all masked
        pixels. Uses caching to prevent redundant expensive computations across UI redraws.
        """
        if self.frame is None:
            return None, {}, "Wait for Frame & Mask"
            
        img_bgr = self.frame
        if img_bgr.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
            
        # 1. Update the heavy mapping cache ONLY if needed and possible
        if self._needs_remapping and self.mask is not None:
            mask_data = self.get_aligned_mask()
            mask_bool = mask_data > 127
            pixels_bgr = img_bgr[mask_bool]
            total_pixels = len(pixels_bgr)

            if total_pixels > 0 and self.palette_lab is not None:
                # Perform HEAVY MAPPING via BLAS SGEMM
                pixels_lab = cv2.cvtColor(pixels_bgr[None, :, :], cv2.COLOR_BGR2Lab)[0].astype(np.float32)
                scores = pixels_lab @ self._palette_lab_T
                scores -= self._palette_half_norm_sq
                closest_indices = np.argmax(scores, axis=1)
                
                mapped_pixels_bgr = self.palette_bgr[closest_indices]
                
                cached_frame = np.zeros_like(img_bgr)
                cached_frame[mask_bool] = mapped_pixels_bgr
                
                index_map = np.full(img_bgr.shape[:2], -1, dtype=np.int16)
                index_map[mask_bool] = closest_indices
                self._cached_index_map = index_map
                
                counts = np.bincount(closest_indices, minlength=len(self.palette_keys))
                cached_dist = {}
                for i, name in enumerate(self.palette_keys):
                    count = counts[i]
                    cached_dist[name] = {
                        "count": int(count),
                        "percent": float(count / total_pixels * 100),
                        "color": [int(c) for c in self.palette_rgb[i]]
                    }
                
                # ... and also cache the MASKED ORIGINAL for the preview mode
                preview_frame = np.zeros_like(img_bgr)
                preview_frame[mask_bool] = pixels_bgr
                
                self._cached_mapped_frame = cached_frame
                self._cached_preview_frame = preview_frame
                
                # Add Best Match
                best_idx = np.argmax(counts)
                cached_dist["best_match"] = self.palette_keys[best_idx]
                
                self._cached_distribution = cached_dist
            else:
                self._cached_mapped_frame = None 
                self._cached_preview_frame = img_bgr.copy()
                self._cached_index_map = None
                self._cached_distribution = {}
            
            self._needs_remapping = False
            msg_suffix = "" 
        else:
            msg_suffix = " (Cached)" if self._cached_mapped_frame is not None else ""

        # 2. Logic for display
        if show_original:
            return img_bgr, {}, "Original View" + msg_suffix
            
        elif is_comparing:
            # Use cached masked original
            if self._cached_preview_frame is not None:
                return self._cached_preview_frame, {}, "Preview Original (Masked)" + msg_suffix
            return img_bgr, {}, "Preview Original" + msg_suffix

        # If we have a cached mapped frame, show it
        if self._cached_mapped_frame is not None:
            return self._cached_mapped_frame, self._cached_distribution, "Ready" + msg_suffix
        
        # Default: just show the original frame
        return img_bgr, {}, "Ready (No mapping calculated)"

    def get_roi_distribution(self, x1: int, y1: int, x2: int, y2: int, use_mask: bool = True) -> dict[str, dict]:
        """
        Calculate palette color distribution specifically within a rectangular Region of Interest (ROI).
        """
        if self.frame is None or self.palette_lab is None:
            return {}
            
        img_bgr = self.frame
        # Ensure coordinates are within bounds
        h, w = img_bgr.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return {}

        # Fast path: reuse cached 2D palette index map when available
        if use_mask and self.mask is not None and not self._needs_remapping and self._cached_index_map is not None:
            idx_roi = self._cached_index_map[y1:y2, x1:x2]
            valid_indices = idx_roi[idx_roi >= 0]
            if valid_indices.size == 0:
                return {}
            counts = np.bincount(valid_indices, minlength=len(self.palette_keys))
            total_pixels = valid_indices.size
            roi_dist = {}
            for i, name in enumerate(self.palette_keys):
                count = counts[i]
                roi_dist[name] = {
                    "count": int(count),
                    "percent": float(count / total_pixels * 100),
                    "color": [int(c) for c in self.palette_rgb[i]]
                }
            roi_dist["best_match"] = self.palette_keys[int(np.argmax(counts))]
            return roi_dist
        
        roi = img_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return {}
            
        if use_mask and self.mask is not None:
            mask_data = self.get_aligned_mask()
            mask_roi = mask_data[y1:y2, x1:x2]
            roi_pixels = roi[mask_roi > 127]
            if roi_pixels.size == 0: return {}
        else:
            roi_pixels = roi.reshape(-1, 3)
        if roi_pixels.size == 0: return {}
        
        try:
            # OpenCV BGR to LAB expects 3 channels. Reshaping to (1, N, 3)
            # Use COLOR_BGR2LAB (all caps) to be sure
            pixels_reshaped = roi_pixels.reshape(1, -1, 3).astype(np.uint8)
            roi_lab = cv2.cvtColor(pixels_reshaped, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
            
            # self.palette_lab is (M, 3). roi_lab is (N, 3)
            # Check broadcasting compatibility
            if self.palette_lab.shape[1] != 3:
                return {}
                
            scores = roi_lab @ self._palette_lab_T
            scores -= self._palette_half_norm_sq
            closest_indices = np.argmax(scores, axis=1)
            
            counts = np.bincount(closest_indices, minlength=len(self.palette_lab))
            
            roi_dist = {}
            total_pixels = len(closest_indices)
            for i, name in enumerate(self.palette_keys):
                count = counts[i]
                roi_dist[name] = {
                    "count": int(count),
                    "percent": float(count / total_pixels * 100) if total_pixels > 0 else 0.0,
                    "color": [int(c) for c in self.palette_rgb[i]]
                }
            
            best_idx = np.argmax(counts)
            roi_dist["best_match"] = self.palette_keys[best_idx] if total_pixels > 0 else "None"
            
            return roi_dist
        except Exception as e:
            # Silently fail here, the window will log the error if needed
            return {}

    def analyze_roi(self, x1: int, y1: int, x2: int, y2: int, use_mask: bool) -> tuple[tuple[int, int, int] | None, str]:
        """
        Analyze the average color within an ROI and find its closest palette match.
        """
        if self.frame is None or self.palette_lab is None:
            return None, ""
            
        roi = self.frame[y1:y2, x1:x2]
        if use_mask and self.mask is not None:
            mask_data = self.get_aligned_mask()
            roi_mask = mask_data[y1:y2, x1:x2]
            valid_pixels = roi[roi_mask > 127]
        else:
            valid_pixels = roi.reshape(-1, 3) if roi.size > 0 else np.array([])
        
        if valid_pixels.size > 0:
            mean_bgr = np.mean(valid_pixels, axis=0)
            mean_rgb = (int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0]))
            
            mean_lab = cv2.cvtColor(np.uint8([[mean_rgb]]), cv2.COLOR_RGB2Lab)[0][0].astype(np.float32)
            diff = mean_lab - self.palette_lab
            dist_sq = np.sum(diff**2, axis=-1)
            closest_idx = np.argmin(dist_sq)
            closest_name = self.palette_keys[closest_idx]
            
            return mean_rgb, closest_name
            
        return None, ""
        
    def calibrate_key_from_roi(self, target_key: str, x1: int, y1: int, x2: int, y2: int, use_mask: bool) -> bool:
        """
        Calibrate an existing palette color key to the mean RGB value of a selected ROI.
        """
        if target_key not in self.color_map_RGB:
            return False
            
        mean_rgb, _ = self.analyze_roi(x1, y1, x2, y2, use_mask)
        if mean_rgb is not None:
            self.color_map_RGB[target_key] = mean_rgb
            self.precompute_palette()
            return True
        return False
