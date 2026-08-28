import re
from typing import List, Tuple, Optional, Union
from modules.nodal_data_processor.data_processor_base import DataProcessorBase


class FvFmProcessor(DataProcessorBase):
    """Compute Fv/Fm = (Fm - F0) / Fm. Auto-detects F0/Fm pairs from labels if available."""

    def __init__(self, node_id: Union[int, str]):
        super().__init__(node_id)

    def process(self, x_list: List[float], y_list: List[float], labels: Optional[List[str]] = None) -> Tuple[List[float], List[float], Optional[List[str]]]:
        """
        Compute (Fm - F0) / Fm for detected pairs.
        
        If labels are provided, auto-detect F0/Fm by name pattern and pair them by timestamp.
        Otherwise, fall back to sequential pairing: (y[0],y[1]), (y[2],y[3]), ...
        """
        if not self.validate_data(x_list, y_list):
            return x_list, y_list, labels

        # Try smart pairing if labels are available
        if labels and len(labels) == len(y_list):
            result = self._process_with_labels(x_list, y_list, labels)
            if result is not None:
                return result

        # Fallback: sequential pairing (original behavior)
        return self._process_sequential(x_list, y_list, labels)

    def _process_with_labels(self, x_list: List[float], y_list: List[float], labels: List[str]) -> Optional[Tuple[List[float], List[float], List[str]]]:
        """
        Smart F0/Fm pairing based on label names.
        
        Expected patterns: 
          - "F0_100", "Fm_200" (prefix + timestamp)
          - "0_GROUP_F0_A0_100" (structured naming)
        
        Returns None if labels don't contain recognizable F0/Fm patterns.
        """
        # Parse each label to extract type (F0/Fm) and timestamp
        f0_data = {}  # {timestamp: (x, y, label)}
        fm_data = {}
        
        # Pattern: look for F0 or Fm anywhere in the label (case insensitive)
        f0_pattern = re.compile(r'(?:^|[_\-])F0(?:[_\-]|$)', re.IGNORECASE)
        fm_pattern = re.compile(r'(?:^|[_\-])Fm(?:[_\-]|$)', re.IGNORECASE)
        
        has_f0 = False
        has_fm = False
        
        for i, label in enumerate(labels):
            name = label.split('.')[0] if '.' in label else label  # Remove extension
            
            is_f0 = bool(f0_pattern.search(name))
            is_fm = bool(fm_pattern.search(name))
            
            if not is_f0 and not is_fm:
                continue
            
            if is_f0:
                has_f0 = True
            if is_fm:
                has_fm = True
                 
            # Extract timestamp: last numeric part after underscore
            parts = name.split('_')
            timestamp = None
            for part in reversed(parts):
                if part.isdigit():
                    timestamp = int(part)
                    break
            
            if timestamp is None:
                timestamp = i  # Fallback: use index as timestamp
            
            if is_f0:
                f0_data[timestamp] = (x_list[i], y_list[i], label)
            elif is_fm:
                fm_data[timestamp] = (x_list[i], y_list[i], label)
        
        # Need both F0 and Fm to do smart pairing
        if not has_f0 or not has_fm:
            return None
        
        # Match F0/Fm pairs by closest timestamp
        out_x = []
        out_y = []
        out_labels = []
        
        # Sort F0 timestamps
        f0_timestamps = sorted(f0_data.keys())
        fm_timestamps = sorted(fm_data.keys())
        
        # Pair: for each F0, find the closest Fm with timestamp >= F0
        used_fm = set()
        for f0_ts in f0_timestamps:
            best_fm_ts = None
            best_dist = float('inf')
            
            for fm_ts in fm_timestamps:
                if fm_ts in used_fm:
                    continue
                dist = abs(fm_ts - f0_ts)
                if dist < best_dist:
                    best_dist = dist
                    best_fm_ts = fm_ts
            
            if best_fm_ts is not None:
                used_fm.add(best_fm_ts)
                
                f0_x, f0_y, f0_label = f0_data[f0_ts]
                fm_x, fm_y, fm_label = fm_data[best_fm_ts]
                
                fvfm = (fm_y - f0_y) / fm_y if fm_y != 0 else 0.0
                out_x.append(fm_x)  # Use Fm's x value
                out_y.append(fvfm)
                out_labels.append(f"{f0_label}+{fm_label}")
        
        if not out_x:
            return None
            
        return out_x, out_y, out_labels

    def _process_sequential(self, x_list: List[float], y_list: List[float], labels: Optional[List[str]]) -> Tuple[List[float], List[float], Optional[List[str]]]:
        """Original sequential pairing: (y[0],y[1]), (y[2],y[3]), ..."""
        out_x = []
        out_y = []
        out_labels = []
        
        for i in range(0, len(y_list) - 1, 2):
            f0 = y_list[i]
            fm = y_list[i + 1]
            fvfm = (fm - f0) / fm if fm != 0 else 0.0
            out_x.append(x_list[i + 1])  # Use the Fm sample's x value
            out_y.append(fvfm)
            
            if labels and i + 1 < len(labels):
                out_labels.append(f"{labels[i]}+{labels[i+1]}")

        result_labels = out_labels if out_labels else labels
        return out_x, out_y, result_labels

