import re
from typing import List, Tuple, Optional, Union
import dearpygui.dearpygui as dpg
from loguru import logger
from modules.nodal_data_processor.data_processor_base import DataProcessorBase


class CorrectF0(DataProcessorBase):
    """Correct F0 values based on detector error equation: % Error = a * detector^b.
    
    Formula:
        Error % = a * (detector ^ b)
        F0_corrected = F0 / (1 + Error% / 100)
    Only F0 images are modified; other points (e.g. Fm, OJIP) remain unchanged.
    """

    def __init__(self, node_id: Union[int, str]):
        super().__init__(node_id)
        # Equation parameters: % Error = a * (detector ^ b)
        self.a: float = 521.8
        self.b: float = -0.885
        self.default_detector: float = 500.0

        self.f0_pattern: re.Pattern = re.compile(r'(?:^|[_\-])F0(?:[_\-]|$)', re.IGNORECASE)
        self.fm_pattern: re.Pattern = re.compile(r'(?:^|[_\-])Fm(?:[_\-]|$)', re.IGNORECASE)

        self.info_text_tag: Optional[str] = None

    def get_current_detector(self) -> float:
        """Retrieve detector from app_state. If None, use default_detector (500)."""
        from core.app_state import app_state
        det = getattr(app_state, "detector", None)
        if det is None:
            return float(self.default_detector)
        try:
            val = float(det)
            return val if val > 0 else float(self.default_detector)
        except (ValueError, TypeError):
            return float(self.default_detector)

    def compute_error_percentage(self, detector: float) -> float:
        """Compute error percentage using Y = a * (detector ^ b)."""
        if detector <= 0:
            return 0.0
        try:
            return float(self.a * (detector ** self.b))
        except Exception as e:
            logger.warning(f"Error computing error percentage: {e}")
            return 0.0

    def correct_f0_value(self, val: float, error_pct: float) -> float:
        """Apply correction to an F0 value: F0_corr = F0 / (1 + error% / 100)."""
        denom = 1.0 + (error_pct / 100.0)
        if denom == 0:
            return val
        return float(val / denom)

    def _is_f0(self, label: Optional[str]) -> bool:
        """Determine whether the point is an F0 measurement."""
        if not label:
            return False
        if "+" in label:
            return False
        name = label.split(".")[0] if "." in label else label
        is_f0 = bool(self.f0_pattern.search(name))
        is_fm = bool(self.fm_pattern.search(name))
        return is_f0 and not is_fm

    def process(
        self,
        x_list: List[float],
        y_list: List[float],
        labels: Optional[List[str]] = None,
    ) -> Tuple[List[float], List[float], Optional[List[str]]]:
        """Recalculate F0 values using the error equation based on detector."""
        if not self.validate_data(x_list, y_list):
            return x_list, y_list, labels

        detector = self.get_current_detector()
        error_pct = self.compute_error_percentage(detector)

        # Update UI info tag if present
        self._update_info_ui(detector, error_pct)

        corrected_y: List[float] = []
        for i, y in enumerate(y_list):
            lbl = labels[i] if (labels and i < len(labels)) else None
            if self._is_f0(lbl):
                corrected_y.append(self.correct_f0_value(y, error_pct))
            else:
                corrected_y.append(y)

        return x_list, corrected_y, labels

    def _update_info_ui(self, detector: float, error_pct: float) -> None:
        """Update UI text showing current error percentage."""
        if self.info_text_tag and dpg.does_item_exist(self.info_text_tag):
            dpg.set_value(self.info_text_tag, f"{error_pct:.2f}%")

    def create_ui(self, parent: Union[int, str]) -> None:
        """Create UI controls for equation parameters and default detector."""
        # Parameter a
        w_a = dpg.add_input_float(
            label="a",
            default_value=self.a,
            step=0.0,
            step_fast=0.0,
            format="%.2f",
            parent=parent,
            width=100,
            callback=lambda s, v: self.update_setting("a", float(v)),
        )
        self.register_param("a", w_a)
        with dpg.tooltip(w_a):
            dpg.add_text("Coefficient 'a' in error formula Y = a * detector^b (default: 521.8).", wrap=250)

        # Parameter b
        w_b = dpg.add_input_float(
            label="b",
            default_value=self.b,
            step=0.0,
            step_fast=0.0,
            format="%.4f",
            parent=parent,
            width=100,
            callback=lambda s, v: self.update_setting("b", float(v)),
        )
        self.register_param("b", w_b)
        with dpg.tooltip(w_b):
            dpg.add_text("Exponent 'b' in error formula Y = a * detector^b (default: -0.885).", wrap=250)

        # Default detector (fallback if app_state.detector is None)
        w_default_det = dpg.add_input_float(
            label="Default Det.",
            default_value=self.default_detector,
            step=0.0,
            step_fast=0.0,
            format="%.1f",
            parent=parent,
            width=100,
            callback=lambda s, v: self.update_setting("default_detector", float(v)),
        )
        self.register_param("default_detector", w_default_det)
        with dpg.tooltip(w_default_det):
            dpg.add_text("Default detector used when app_state.detector is None (default: 500).", wrap=250)

        # Live info text
        self.info_text_tag = f"correct_f0_info_{self.uuid}"
        detector = self.get_current_detector()
        error_pct = self.compute_error_percentage(detector)
        dpg.add_text(
            f"{error_pct:.2f}% actuel",
            tag=self.info_text_tag,
            parent=parent,
            color=(130, 220, 150),
        )


# Alias
CorrectF0Processor = CorrectF0
