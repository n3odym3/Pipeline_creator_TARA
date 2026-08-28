import re
from typing import List, Tuple, Optional, Union, Dict
from pathlib import Path
from modules.nodal_data_processor.data_processor_base import DataProcessorBase


class RETR(DataProcessorBase):
    """
    Compute rETR = (Fv/Fm) * Actinic Intensity.
    Formula: [(Fm - F0) / Fm] * Intensity
    Intensity is extracted from labels (e.g. A100) or fallback to X value.
    """

    def __init__(self, node_id: Union[int, str]):
        super().__init__(node_id)
        # Pairing patterns for F0 and Fm
        self.f0_pattern: re.Pattern = re.compile(r'(?:^|[_\-])F0(?:[_\-]|$)', re.IGNORECASE)
        self.fm_pattern: re.Pattern = re.compile(r'(?:^|[_\-])Fm(?:[_\-]|$)', re.IGNORECASE)

    def _get_intensity(self, label: str, x_val: float) -> float:
        """Extract intensity from label or fallback to X value."""
        if not label:
            return x_val

        filename = label.split('+')[0]
        name_no_ext = Path(filename).stem
    
        a_match = re.search(r'[_\-]A(\d+)(?:[_\-]|$)', f"_{name_no_ext}_", re.IGNORECASE)
        if a_match:
            try:
                return float(a_match.group(1))
            except (ValueError, IndexError):
                pass
            
        return x_val

    def process(self, x_list: List[float], y_list: List[float], labels: Optional[List[str]] = None) -> Tuple[List[float], List[float], Optional[List[str]]]:
        """Entry point for nodal processor."""
        if not self.validate_data(x_list, y_list):
            return x_list, y_list, labels

        if labels and len(labels) == len(y_list):
            result = self._process_with_labels(x_list, y_list, labels)
            if result is not None:
                return result

        return self._process_sequential(x_list, y_list, labels)

    def _process_with_labels(self, x_list: List[float], y_list: List[float], labels: List[str]) -> Optional[Tuple[List[float], List[float], List[str]]]:
        """Smart F0/Fm pairing + intensity extraction."""
        f0_data: Dict[int, Tuple[float, float, str]] = {}  # {timestamp: (x, y, label)}
        fm_data: Dict[int, Tuple[float, float, str]] = {}
        
        for i, label in enumerate(labels):
            name = Path(label).stem
            is_f0 = bool(self.f0_pattern.search(name))
            is_fm = bool(self.fm_pattern.search(name))
            
            if not is_f0 and not is_fm:
                continue
            
            # Extract timestamp: last numeric part after underscore
            parts = name.split('_')
            timestamp = i # fallback
            for part in reversed(parts):
                if part.isdigit():
                    timestamp = int(part)
                    break
            
            if is_f0:
                f0_data[timestamp] = (x_list[i], y_list[i], label)
            elif is_fm:
                fm_data[timestamp] = (x_list[i], y_list[i], label)
        
        if not f0_data or not fm_data:
            return None
        
        out_x, out_y, out_labels = [], [], []
        f0_timestamps = sorted(f0_data.keys())
        fm_timestamps = sorted(fm_data.keys())
        
        used_fm = set()
        for f0_ts in f0_timestamps:
            best_fm_ts = None
            best_dist = float('inf')
            
            for fm_ts in fm_timestamps:
                if fm_ts in used_fm: continue
                dist = abs(fm_ts - f0_ts)
                if dist < best_dist:
                    best_dist = dist
                    best_fm_ts = fm_ts
            
            if best_fm_ts is not None:
                used_fm.add(best_fm_ts)
                f0_x, f0_y, f0_label = f0_data[f0_ts]
                fm_x, fm_y, fm_label = fm_data[best_fm_ts]
                
                # Formula: [ (Fm - F0) / Fm ] * Intensity
                if fm_y != 0:
                    fvfm = (fm_y - f0_y) / fm_y
                    intensity = self._get_intensity(f0_label, f0_x)
                    retr = fvfm * intensity
                    out_x.append(intensity)
                    out_y.append(float(retr))
                    out_labels.append(f"{f0_label}+{fm_label}")
        
        return out_x, out_y, out_labels if out_labels else None

    def _process_sequential(self, x_list: List[float], y_list: List[float], labels: Optional[List[str]]) -> Tuple[List[float], List[float], Optional[List[str]]]:
        """Sequential fallback pairing: (y[0],y[1]), (y[2],y[3]), ..."""
        out_x, out_y, out_labels = [], [], []
        
        for i in range(0, len(y_list) - 1, 2):
            f0_y, fm_y = y_list[i], y_list[i + 1]
            if fm_y != 0:
                fvfm = (fm_y - f0_y) / fm_y
                f0_label = labels[i] if (labels and i < len(labels)) else ""
                intensity = self._get_intensity(f0_label, x_list[i])
                retr = fvfm * intensity
                
                out_x.append(intensity)
                out_y.append(float(retr))
                if labels and i+1 < len(labels):
                    out_labels.append(f"{labels[i]}+{labels[i+1]}")

        return out_x, out_y, out_labels if out_labels else labels

