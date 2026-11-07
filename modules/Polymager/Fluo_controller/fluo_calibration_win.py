from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional
import dearpygui.dearpygui as dpg
from loguru import logger
import numpy as np

from config.display_scaling import display_scaling
from core.file_explorer import file_explorer
from .serial_control import SerialControl


class CalibrationModelFit:
    MODEL_CHOICES = ["Linear (a * x + b)", "Polynomial", "Power Law (a * x^b)"]
    MODEL_TYPE_TO_LABEL = {
        "linear": "Linear (a * x + b)",
        "polynomial": "Polynomial",
        "power": "Power Law (a * x^b)",
    }
    LABEL_TO_MODEL_TYPE = {
        "Linear (a * x + b)": "linear",
        "Polynomial": "polynomial",
        "Power Law (a * x^b)": "power",
    }

    @staticmethod
    def fit(
        x_data: List[float],
        y_data: List[float],
        model_type: str = "linear",
        poly_degree: int = 2,
    ) -> tuple[bool, str, float, Optional[Callable[[Any], Any]], Optional[Callable[[Any], Any]], dict]:
        """
        Fits a regression model mapping hardware command x (0..512) -> measured intensity y (µE).
        Returns:
            is_valid (bool), eq_str (str), r2 (float), forward_fn (x->y), inverse_fn (y->x), params (dict)
        """
        if len(x_data) < 2 or len(y_data) < 2:
            return False, "Need at least 2 points for fitting", 0.0, None, None, {}

        x_arr = np.asarray(x_data, dtype=float)
        y_arr = np.asarray(y_data, dtype=float)

        try:
            if model_type == "power":
                mask = (x_arr > 0) & (y_arr > 0)
                if np.sum(mask) < 2:
                    return False, "Need positive values for power-law fit", 0.0, None, None, {}
                log_x = np.log(x_arr[mask])
                log_y = np.log(y_arr[mask])
                b, log_a = np.polyfit(log_x, log_y, 1)
                a = float(np.exp(log_a))
                b = float(b)

                def forward_fn(x: Any) -> Any:
                    return a * np.power(np.maximum(0.0, np.asarray(x, dtype=float)), b)

                def inverse_fn(y: Any) -> Any:
                    return np.power(np.maximum(1e-9, np.asarray(y, dtype=float)) / max(1e-9, a), 1.0 / max(1e-9, b))

                eq_str = f"I = {a:.4f} × X^{b:.4f}"
                params = {"type": "power", "a": a, "b": b}

            elif model_type == "exponential":
                mask = (y_arr > 0)
                if np.sum(mask) < 2:
                    return False, "Need positive intensity for exponential fit", 0.0, None, None, {}
                log_y = np.log(y_arr[mask])
                b, log_a = np.polyfit(x_arr[mask], log_y, 1)
                a = float(np.exp(log_a))
                b = float(b)

                def forward_fn(x: Any) -> Any:
                    return a * np.exp(b * np.asarray(x, dtype=float))

                def inverse_fn(y: Any) -> Any:
                    denom = b if abs(b) > 1e-9 else 1.0
                    return np.log(np.maximum(1e-9, np.asarray(y, dtype=float)) / max(1e-9, a)) / denom

                eq_str = f"I = {a:.4f} × exp({b:.4f} × X)"
                params = {"type": "exponential", "a": a, "b": b}

            elif model_type == "polynomial":
                deg = min(max(2, int(poly_degree)), min(5, len(x_arr) - 1))
                coeffs = [float(c) for c in np.polyfit(x_arr, y_arr, deg)]

                x_min = 0.0
                x_max = float(np.max(x_arr))
                if x_max > 400.0:
                    x_max = max(512.0, x_max)

                grid_x = np.linspace(x_min, x_max, 2048)
                grid_y = np.polyval(coeffs, grid_x)

                # Ensure strict monotonicity for np.interp inversion
                grid_y_mono = np.maximum.accumulate(grid_y)
                for i in range(1, len(grid_y_mono)):
                    if grid_y_mono[i] <= grid_y_mono[i - 1]:
                        grid_y_mono[i] = grid_y_mono[i - 1] + 1e-7

                def forward_fn(x: Any) -> Any:
                    return np.polyval(coeffs, np.asarray(x, dtype=float))

                def inverse_fn(y: Any) -> Any:
                    return np.interp(np.asarray(y, dtype=float), grid_y_mono, grid_x, left=x_min, right=x_max)

                terms = []
                for i, c in enumerate(coeffs):
                    power = deg - i
                    sign = "+" if (c >= 0 and i > 0) else ("-" if c < 0 and i > 0 else "")
                    val = abs(c) if i > 0 else c
                    if power == 0:
                        terms.append(f"{sign} {val:.2f}" if i > 0 else f"{val:.2f}")
                    elif power == 1:
                        terms.append(f"{sign} {val:.4f}×X" if i > 0 else f"{val:.4f}×X")
                    elif power == 2:
                        terms.append(f"{sign} {val:.4e}×X²" if i > 0 else f"{val:.4e}×X²")
                    else:
                        terms.append(f"{sign} {val:.4e}×X^{power}" if i > 0 else f"{val:.4e}×X^{power}")
                eq_str = "I = " + " ".join(terms)
                params = {"type": "polynomial", "degree": deg, "coeffs": coeffs}

            else:  # "linear" default
                a, b = np.polyfit(x_arr, y_arr, 1)
                a, b = float(a), float(b)

                def forward_fn(x: Any) -> Any:
                    return a * np.asarray(x, dtype=float) + b

                def inverse_fn(y: Any) -> Any:
                    y_a = np.asarray(y, dtype=float)
                    return (y_a - b) / a if abs(a) > 1e-9 else np.zeros_like(y_a)

                sign = "+" if b >= 0 else "-"
                eq_str = f"I = {a:.4f} × X {sign} {abs(b):.2f}"
                params = {"type": "linear", "a": a, "b": b}

            # Calculate R^2
            y_pred = forward_fn(x_arr)
            ss_res = np.sum((y_arr - y_pred) ** 2)
            ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
            r2 = max(0.0, min(1.0, 1.0 - (ss_res / ss_tot))) if ss_tot > 0 else 1.0

            return True, eq_str, r2, forward_fn, inverse_fn, params

        except Exception as e:
            logger.debug(f"Fit error for {model_type}: {e}")
            return False, f"Fit error: {e}", 0.0, None, None, {}


class FluoCalibrationWin:
    """
    FluoCalibrationWin:
    Interactive calibration wizard for the Fluo Controller Actinic LED.
    Allows automated step-by-step recording of µE intensities at different
    PWM (low regime) and ADIM (high regime) levels with a momentary TEST button,
    table review, configurable regression models, and JSON export/import.
    """

    DEFAULT_LOW_PWM = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
    DEFAULT_HIGH_ADIM = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 512]

    def __init__(
        self,
        serial: SerialControl,
        on_calibration_saved: Optional[Callable[[str], None]] = None,
        parent_uuid: Optional[int | str] = None,
        initial_cal_path: Optional[str] = None,
        initial_cal_data: Optional[dict] = None,
        show: bool = True,
    ) -> None:
        self.serial = serial
        self.on_calibration_saved = on_calibration_saved
        self.uuid = parent_uuid or "calib"
        self.loaded_file_path: Optional[str] = None

        # Calibration Data State
        self.threshold: int = 150
        self.low_cal_steps: List[int] = list(self.DEFAULT_LOW_PWM)
        self.low_cal_data: Dict[int, Optional[float]] = {pwm: None for pwm in self.low_cal_steps}

        self.high_cal_steps: List[int] = list(self.DEFAULT_HIGH_ADIM)
        self.high_cal_data: Dict[int, Optional[float]] = {adim: None for adim in self.high_cal_steps}

        self._built_low_steps: List[int] = []
        self._built_high_steps: List[int] = []

        # Model Regression Selection
        self.low_model_type: str = "power"  # "power", "linear", "exponential", "polynomial"
        self.low_poly_degree: int = 2
        self.high_model_type: str = "linear"  # "linear", "power", "exponential", "polynomial"
        self.high_poly_degree: int = 2

        self.current_mode: str = "low_cal"  # "low_cal" or "high_cal"
        self.current_step_idx: int = 0
        self._is_testing: bool = False

        # If no initial_cal_path given, check default fluo_calibration.json
        if initial_cal_path is None:
            default_path = os.path.join(os.path.dirname(__file__), "fluo_calibration.json")
            if os.path.exists(default_path):
                initial_cal_path = default_path

        # Populate internal state from initial_cal_data or initial_cal_path if present
        if initial_cal_data is not None:
            self._apply_calibration_dict(initial_cal_data, file_path=initial_cal_path)
        elif initial_cal_path and os.path.exists(initial_cal_path):
            try:
                with open(initial_cal_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._apply_calibration_dict(data, file_path=initial_cal_path)
            except Exception as e:
                logger.warning(f"Failed to load initial calibration from {initial_cal_path}: {e}")

        # Tag definitions
        self.win_tag = f"fluo_calib_win_{self.uuid}"
        self.thresh_input_tag = f"fluo_calib_thresh_{self.uuid}"
        self.active_param_text_tag = f"fluo_calib_active_param_{self.uuid}"
        self.measured_ue_tag = f"fluo_calib_measured_ue_{self.uuid}"
        self.btn_test_tag = f"fluo_calib_btn_test_{self.uuid}"
        self.btn_record_tag = f"fluo_calib_btn_record_{self.uuid}"
        self.test_handler_tag = f"fluo_calib_test_handler_{self.uuid}"
        self.global_handler_tag = f"fluo_calib_global_handler_{self.uuid}"
        self.summary_text_tag = f"fluo_calib_summary_{self.uuid}"
        self.tab_bar_tag = f"fluo_calib_tabs_{self.uuid}"
        self.tab_low_tag = f"fluo_calib_tab_low_{self.uuid}"
        self.tab_high_tag = f"fluo_calib_tab_high_{self.uuid}"
        self.tab_curves_tag = f"fluo_calib_tab_curves_{self.uuid}"
        self.low_table_tag = f"fluo_calib_low_table_{self.uuid}"
        self.high_table_tag = f"fluo_calib_high_table_{self.uuid}"
        self.status_msg_tag = f"fluo_calib_status_msg_{self.uuid}"

        # Model Plot tags
        self.comb_plot_tag = f"fluo_calib_comb_plot_{self.uuid}"
        self.comb_xaxis_tag = f"fluo_calib_comb_xaxis_{self.uuid}"
        self.comb_yaxis_tag = f"fluo_calib_comb_yaxis_{self.uuid}"
        self.comb_model_text_tag = f"fluo_calib_comb_model_text_{self.uuid}"
        self.drag_line_tag = f"fluo_calib_drag_line_{self.uuid}"
        self.adjust_checkbox_tag = f"fluo_calib_adjust_check_{self.uuid}"

        self.low_plot_tag = f"fluo_calib_low_plot_{self.uuid}"
        self.low_xaxis_tag = f"fluo_calib_low_xaxis_{self.uuid}"
        self.low_yaxis_tag = f"fluo_calib_low_yaxis_{self.uuid}"
        self.low_model_text_tag = f"fluo_calib_low_model_text_{self.uuid}"
        self.low_model_combo_tag = f"fluo_calib_low_model_combo_{self.uuid}"
        self.low_degree_label_tag = f"fluo_calib_low_deg_lbl_{self.uuid}"
        self.low_degree_input_tag = f"fluo_calib_low_deg_inp_{self.uuid}"

        self.high_plot_tag = f"fluo_calib_high_plot_{self.uuid}"
        self.high_xaxis_tag = f"fluo_calib_high_xaxis_{self.uuid}"
        self.high_yaxis_tag = f"fluo_calib_high_yaxis_{self.uuid}"
        self.high_model_text_tag = f"fluo_calib_high_model_text_{self.uuid}"
        self.high_model_combo_tag = f"fluo_calib_high_model_combo_{self.uuid}"
        self.high_degree_label_tag = f"fluo_calib_high_deg_lbl_{self.uuid}"
        self.high_degree_input_tag = f"fluo_calib_high_deg_inp_{self.uuid}"

        if show:
            self._build_ui(show=True)

    def show(self) -> None:
        """Show and bring the calibration window to the front."""
        if dpg.does_item_exist(self.win_tag):
            dpg.show_item(self.win_tag)
            dpg.focus_item(self.win_tag)
            self._refresh_tables()
            self._set_active_step(self.current_mode, self.current_step_idx)
        else:
            self._build_ui(show=True)
            dpg.focus_item(self.win_tag)

    def _build_ui(self, show: bool = True) -> None:
        s = display_scaling.scale
        self._built_low_steps = []
        self._built_high_steps = []

        if dpg.does_item_exist(self.win_tag):
            dpg.delete_item(self.win_tag)

        # Cleanup existing handlers if any
        if dpg.does_item_exist(self.test_handler_tag):
            dpg.delete_item(self.test_handler_tag)
        if dpg.does_item_exist(self.global_handler_tag):
            dpg.delete_item(self.global_handler_tag)
        if dpg.does_item_exist(self.drag_line_tag):
            dpg.delete_item(self.drag_line_tag)

        with dpg.window(
            label="Fluorescence Actinic Calibration Tool",
            tag=self.win_tag,
            width=s(620),
            height=s(700),
            pos=(s(100), s(80)),
            on_close=self.on_close,
            show=show,
        ):
            # Top Header: Title & Threshold
            with dpg.group(horizontal=True):
                dpg.add_text("Actinic LED Calibration Assistant", color=(100, 200, 255))
                dpg.add_spacer(width=s(25))
                dpg.add_text("Threshold (µE):")
                dpg.add_input_int(
                    tag=self.thresh_input_tag,
                    default_value=self.threshold,
                    width=s(110),
                    min_value=0,
                    max_value=5000,
                    step=-1,
                    callback=self._on_thresh_input_changed,
                )
                with dpg.tooltip(self.thresh_input_tag):
                    dpg.add_text(
                        "Intensity threshold (µE) separating the two regimes:\n"
                        "• <= Threshold: Low Mode (PWM variable, ADIM=512)\n"
                        "• > Threshold: High Mode (ADIM variable, PWM=512)",
                        wrap=s(350),
                    )

            dpg.add_spacer(height=s(2))

            # Active Setting text directly in window
            dpg.add_text("Active Setting: PWM = 5, ADIM = 512", tag=self.active_param_text_tag, color=(255, 215, 0))

            dpg.add_spacer(height=s(3))

            # 1. TEST (Hold) on its own line
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="TEST (Hold)",
                    tag=self.btn_test_tag,
                    width=s(140),
                    height=s(30),
                )
                with dpg.tooltip(self.btn_test_tag):
                    dpg.add_text(
                        "Hold down to turn ON the Actinic LED at current step setting.\n"
                        "Release mouse to turn OFF immediately.",
                        wrap=s(320),
                    )
                dpg.add_spacer(width=s(10))
                status_text = f"Loaded: {os.path.basename(self.loaded_file_path)}" if self.loaded_file_path else "Status: OFF"
                status_color = (100, 255, 100) if self.loaded_file_path else (160, 160, 160)
                dpg.add_text(status_text, tag=self.status_msg_tag, color=status_color)

            dpg.add_spacer(height=s(3))

            # 2. Below: Measured (µE) & Record & Next
            with dpg.group(horizontal=True):
                dpg.add_text("Measured (µE):")
                dpg.add_input_float(
                    tag=self.measured_ue_tag,
                    default_value=0.0,
                    format="%.2f",
                    width=s(120),
                    step=-1,
                )
                dpg.add_spacer(width=s(8))
                dpg.add_button(
                    label="Record & Next",
                    tag=self.btn_record_tag,
                    width=s(140),
                    height=s(28),
                    callback=self._on_record_and_next,
                )
                with dpg.tooltip(self.btn_record_tag):
                    dpg.add_text(
                        "Record the measured intensity for this step\n"
                        "and advance to the next step.",
                        wrap=s(320),
                    )

            dpg.add_spacer(height=s(3))

            # 3. Below: Prev & Next
            with dpg.group(horizontal=True):
                dpg.add_button(label="< Prev", width=s(120), height=s(28), callback=self._on_prev_step)
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("Previous calibration step (switches to Low Mode if at start of High Mode).")

                dpg.add_spacer(width=s(10))
                dpg.add_button(label="Next >", width=s(120), height=s(28), callback=self._on_next_step)
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("Next calibration step (switches to High Mode if at end of Low Mode).")

            dpg.add_spacer(height=s(2))

            # Progress Summary
            dpg.add_text("Progress: Low Mode 0/15 points | High Mode 0/11 points", tag=self.summary_text_tag)

            dpg.add_separator()

            # Dynamic Tables Container that fills remaining vertical space
            with dpg.child_window(height=-s(48), border=False):
                with dpg.tab_bar(tag=self.tab_bar_tag, callback=self._on_tab_changed):
                    with dpg.tab(label="Low Mode Data (PWM)", tag=self.tab_low_tag):
                        with dpg.table(
                            header_row=True,
                            borders_innerH=True,
                            borders_outerH=True,
                            borders_innerV=True,
                            borders_outerV=True,
                            row_background=True,
                            scrollY=True,
                            resizable=True,
                            policy=dpg.mvTable_SizingStretchProp,
                            height=-1,
                            tag=self.low_table_tag,
                        ):
                            dpg.add_table_column(label="Step #", width_stretch=True, init_width_or_weight=0.18)
                            dpg.add_table_column(label="PWM", width_stretch=True, init_width_or_weight=0.22)
                            dpg.add_table_column(label="ADIM (Fixed)", width_stretch=True, init_width_or_weight=0.28)
                            dpg.add_table_column(label="Measured (µE)", width_stretch=True, init_width_or_weight=0.32)

                    with dpg.tab(label="High Mode Data (ADIM)", tag=self.tab_high_tag):
                        with dpg.table(
                            header_row=True,
                            borders_innerH=True,
                            borders_outerH=True,
                            borders_innerV=True,
                            borders_outerV=True,
                            row_background=True,
                            scrollY=True,
                            resizable=True,
                            policy=dpg.mvTable_SizingStretchProp,
                            height=-1,
                            tag=self.high_table_tag,
                        ):
                            dpg.add_table_column(label="Step #", width_stretch=True, init_width_or_weight=0.18)
                            dpg.add_table_column(label="ADIM", width_stretch=True, init_width_or_weight=0.22)
                            dpg.add_table_column(label="PWM (Fixed)", width_stretch=True, init_width_or_weight=0.28)
                            dpg.add_table_column(label="Measured (µE)", width_stretch=True, init_width_or_weight=0.32)

                    with dpg.tab(label="Model / Live Curves", tag=self.tab_curves_tag):
                        with dpg.tab_bar():
                            with dpg.tab(label="Combined Model (All)"):
                                with dpg.group(horizontal=True):
                                    dpg.add_checkbox(
                                        label="Adjust (Align High to Low)",
                                        tag=self.adjust_checkbox_tag,
                                        default_value=False,
                                        callback=lambda: self._refresh_plots(),
                                    )
                                    with dpg.tooltip(self.adjust_checkbox_tag):
                                        dpg.add_text(
                                            "Visual alignment tool:\n"
                                            "Measures the distance between High and Low points in the overlap zone\n"
                                            "and applies a global Y-shift to all High points and the High curve.\n"
                                            "This shows a continuous transition and helps pinpoint exactly\n"
                                            "where the two regimes diverge to choose the ideal threshold.\n"
                                            "(Purely visual tool - does not alter raw calibration values).",
                                            wrap=s(360),
                                        )
                                    dpg.add_spacer(width=s(10))
                                    dpg.add_text(f"Threshold: {self.threshold} µE", tag=self.comb_model_text_tag, color=(255, 215, 0))
                                with dpg.plot(label="Combined Model (Low Power <= Threshold <= High Linear)", height=-1, width=-1, no_menus=True, tag=self.comb_plot_tag):
                                    dpg.add_plot_legend()
                                    dpg.add_plot_axis(dpg.mvXAxis, label="Target Intensity (µE)", tag=self.comb_xaxis_tag)
                                    dpg.add_plot_axis(dpg.mvYAxis, label="Channel Output (0 - 100%)", tag=self.comb_yaxis_tag)
                                    dpg.add_drag_line(
                                        default_value=float(self.threshold),
                                        tag=self.drag_line_tag,
                                        label="Threshold",
                                        color=[255, 60, 60, 255],
                                        thickness=2,
                                        callback=self._on_drag_line_changed,
                                    )

                            with dpg.tab(label="Low Regime (PWM Fit)"):
                                with dpg.group(horizontal=True):
                                    dpg.add_text("Model:")
                                    dpg.add_combo(
                                        items=CalibrationModelFit.MODEL_CHOICES,
                                        default_value=CalibrationModelFit.MODEL_TYPE_TO_LABEL.get(self.low_model_type, "Power Law (a * x^b)"),
                                        tag=self.low_model_combo_tag,
                                        width=s(190),
                                        callback=self._on_low_model_changed,
                                    )
                                    dpg.add_text("Degree:", tag=self.low_degree_label_tag, show=(self.low_model_type == "polynomial"))
                                    dpg.add_input_int(
                                        tag=self.low_degree_input_tag,
                                        default_value=self.low_poly_degree,
                                        min_value=2,
                                        max_value=5,
                                        width=s(100),
                                        step=1,
                                        show=(self.low_model_type == "polynomial"),
                                        callback=self._on_low_degree_changed,
                                    )
                                dpg.add_text("R² = ---", tag=self.low_model_text_tag, color=(255, 200, 100))
                                with dpg.plot(label="Low Regime Curve (PWM vs µE)", height=-1, width=-1, no_menus=True, tag=self.low_plot_tag):
                                    dpg.add_plot_legend()
                                    dpg.add_plot_axis(dpg.mvXAxis, label="PWM (0 - 512)", tag=self.low_xaxis_tag)
                                    dpg.add_plot_axis(dpg.mvYAxis, label="Intensity (µE)", tag=self.low_yaxis_tag)

                            with dpg.tab(label="High Regime (ADIM Fit)"):
                                with dpg.group(horizontal=True):
                                    dpg.add_text("Model:")
                                    dpg.add_combo(
                                        items=CalibrationModelFit.MODEL_CHOICES,
                                        default_value=CalibrationModelFit.MODEL_TYPE_TO_LABEL.get(self.high_model_type, "Linear (a * x + b)"),
                                        tag=self.high_model_combo_tag,
                                        width=s(190),
                                        callback=self._on_high_model_changed,
                                    )
                                    dpg.add_text("Degree:", tag=self.high_degree_label_tag, show=(self.high_model_type == "polynomial"))
                                    dpg.add_input_int(
                                        tag=self.high_degree_input_tag,
                                        default_value=self.high_poly_degree,
                                        min_value=2,
                                        max_value=5,
                                        width=s(100),
                                        step=1,
                                        show=(self.high_model_type == "polynomial"),
                                        callback=self._on_high_degree_changed,
                                    )
                                dpg.add_text("R² = ---", tag=self.high_model_text_tag, color=(255, 200, 100))
                                with dpg.plot(label="High Regime Curve (ADIM vs µE)", height=-1, width=-1, no_menus=True, tag=self.high_plot_tag):
                                    dpg.add_plot_legend()
                                    dpg.add_plot_axis(dpg.mvXAxis, label="ADIM (0 - 512)", tag=self.high_xaxis_tag)
                                    dpg.add_plot_axis(dpg.mvYAxis, label="Intensity (µE)", tag=self.high_yaxis_tag)

            dpg.add_separator()

            # Docked Action Buttons at the bottom
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save", callback=self._on_save_calibration, width=s(150), height=s(30))
                dpg.add_button(label="Load", callback=self._on_load_calibration, width=s(150), height=s(30))
                dpg.add_button(label="Reset", callback=self._on_reset_defaults, width=s(150), height=s(30))
                dpg.add_button(label="Close", callback=lambda: self.on_close(), width=s(150), height=s(30))

        # Setup press-and-hold handler on TEST button
        with dpg.item_handler_registry(tag=self.test_handler_tag):
            dpg.add_item_clicked_handler(callback=self._on_test_press)
        dpg.bind_item_handler_registry(self.btn_test_tag, self.test_handler_tag)

        # Global mouse release handler to ensure LED is turned OFF safely
        with dpg.handler_registry(tag=self.global_handler_tag):
            dpg.add_mouse_release_handler(callback=self._on_test_release)

        self._refresh_tables()
        self._set_active_step(self.current_mode, self.current_step_idx)

    def _get_active_step_value(self) -> int:
        """Returns the current step value (PWM or ADIM) depending on mode."""
        if self.current_mode == "low_cal":
            if 0 <= self.current_step_idx < len(self.low_cal_steps):
                return self.low_cal_steps[self.current_step_idx]
            return 5
        else:
            if 0 <= self.current_step_idx < len(self.high_cal_steps):
                return self.high_cal_steps[self.current_step_idx]
            return 10

    def _set_active_step(self, mode: str, idx: int) -> None:
        """Sets active mode and step index, synchronizes row highlights, tabs, and step UI."""
        self.current_mode = mode
        max_len = len(self.low_cal_steps) if mode == "low_cal" else len(self.high_cal_steps)
        self.current_step_idx = max(0, min(idx, max_len - 1))

        # Synchronize tab if currently on one of the table tabs
        if dpg.does_item_exist(self.tab_bar_tag):
            target_tab = self.tab_low_tag if mode == "low_cal" else self.tab_high_tag
            current_tab = dpg.get_value(self.tab_bar_tag)
            if current_tab in (self.tab_low_tag, self.tab_high_tag) and current_tab != target_tab:
                dpg.set_value(self.tab_bar_tag, target_tab)

        # Update selectable row highlights in both tables
        for j in range(len(self.low_cal_steps)):
            sel_tag = f"low_sel_{self.uuid}_{j}"
            if dpg.does_item_exist(sel_tag):
                dpg.set_value(sel_tag, (mode == "low_cal" and j == self.current_step_idx))

        for j in range(len(self.high_cal_steps)):
            sel_tag = f"high_sel_{self.uuid}_{j}"
            if dpg.does_item_exist(sel_tag):
                dpg.set_value(sel_tag, (mode == "high_cal" and j == self.current_step_idx))

        self._update_step_ui()

    def _on_row_clicked(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Called when a table row is clicked."""
        if not user_data:
            return
        mode, idx = user_data
        self._set_active_step(mode, idx)

    def _on_tab_changed(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Called when user switches tabs in the tab bar."""
        if app_data == self.tab_low_tag:
            if self.current_mode != "low_cal":
                idx = min(self.current_step_idx, len(self.low_cal_steps) - 1)
                self._set_active_step("low_cal", idx)
        elif app_data == self.tab_high_tag:
            if self.current_mode != "high_cal":
                idx = min(self.current_step_idx, len(self.high_cal_steps) - 1)
                self._set_active_step("high_cal", idx)
        elif app_data == self.tab_curves_tag:
            self._refresh_plots()

    def _update_step_ui(self) -> None:
        """Synchronizes active setting text and float input with current step."""
        if self.current_mode == "low_cal":
            steps = self.low_cal_steps
            data = self.low_cal_data
            if self.current_step_idx >= len(steps):
                self.current_step_idx = max(0, len(steps) - 1)

            current_pwm = steps[self.current_step_idx] if steps else 0
            val = data.get(current_pwm)

            if dpg.does_item_exist(self.active_param_text_tag):
                dpg.set_value(
                    self.active_param_text_tag,
                    f"Active Setting: Low Mode (#{self.current_step_idx + 1}/{len(steps)}) — PWM = {current_pwm}, ADIM = 512",
                )

            # Pre-fill input float with recorded value or default 0.0
            if dpg.does_item_exist(self.measured_ue_tag):
                dpg.set_value(self.measured_ue_tag, float(val) if val is not None else 0.0)

        else:
            steps = self.high_cal_steps
            data = self.high_cal_data
            if self.current_step_idx >= len(steps):
                self.current_step_idx = max(0, len(steps) - 1)

            current_adim = steps[self.current_step_idx] if steps else 0
            val = data.get(current_adim)

            if dpg.does_item_exist(self.active_param_text_tag):
                dpg.set_value(
                    self.active_param_text_tag,
                    f"Active Setting: High Mode (#{self.current_step_idx + 1}/{len(steps)}) — ADIM = {current_adim}, PWM = 512",
                )

            if dpg.does_item_exist(self.measured_ue_tag):
                dpg.set_value(self.measured_ue_tag, float(val) if val is not None else 0.0)

        self._update_summary()

    def _update_summary(self) -> None:
        """Updates progress summary text."""
        low_count = sum(1 for v in self.low_cal_data.values() if v is not None)
        high_count = sum(1 for v in self.high_cal_data.values() if v is not None)
        if dpg.does_item_exist(self.summary_text_tag):
            dpg.set_value(
                self.summary_text_tag,
                f"Progress: Low Mode {low_count}/{len(self.low_cal_steps)} points | "
                f"High Mode {high_count}/{len(self.high_cal_steps)} points",
            )

    def _refresh_tables(self) -> None:
        """Repopulates or updates the data tables for low and high calibration regimes with clickable selectable rows."""
        # Low table
        if dpg.does_item_exist(self.low_table_tag):
            if self._built_low_steps != self.low_cal_steps:
                children = dpg.get_item_children(self.low_table_tag, 1)
                if children:
                    for child in children:
                        dpg.delete_item(child)

                for i, pwm in enumerate(self.low_cal_steps):
                    val = self.low_cal_data.get(pwm)
                    val_str = f"{val:.2f}" if val is not None else "---"
                    is_active = (self.current_mode == "low_cal" and self.current_step_idx == i)
                    sel_tag = f"low_sel_{self.uuid}_{i}"
                    val_tag = f"low_val_{self.uuid}_{i}"
                    with dpg.table_row(parent=self.low_table_tag):
                        dpg.add_selectable(
                            label=f"#{i+1}",
                            default_value=is_active,
                            tag=sel_tag,
                            span_columns=True,
                            callback=self._on_row_clicked,
                            user_data=("low_cal", i),
                        )
                        dpg.add_text(str(pwm))
                        dpg.add_text("512")
                        dpg.add_text(val_str, tag=val_tag, color=(100, 255, 100) if val is not None else (150, 150, 150))
                self._built_low_steps = list(self.low_cal_steps)
            else:
                for i, pwm in enumerate(self.low_cal_steps):
                    val = self.low_cal_data.get(pwm)
                    val_str = f"{val:.2f}" if val is not None else "---"
                    val_tag = f"low_val_{self.uuid}_{i}"
                    if dpg.does_item_exist(val_tag):
                        dpg.set_value(val_tag, val_str)
                        dpg.configure_item(val_tag, color=(100, 255, 100) if val is not None else (150, 150, 150))

        # High table
        if dpg.does_item_exist(self.high_table_tag):
            if self._built_high_steps != self.high_cal_steps:
                children = dpg.get_item_children(self.high_table_tag, 1)
                if children:
                    for child in children:
                        dpg.delete_item(child)

                for i, adim in enumerate(self.high_cal_steps):
                    val = self.high_cal_data.get(adim)
                    val_str = f"{val:.2f}" if val is not None else "---"
                    is_active = (self.current_mode == "high_cal" and self.current_step_idx == i)
                    sel_tag = f"high_sel_{self.uuid}_{i}"
                    val_tag = f"high_val_{self.uuid}_{i}"
                    with dpg.table_row(parent=self.high_table_tag):
                        dpg.add_selectable(
                            label=f"#{i+1}",
                            default_value=is_active,
                            tag=sel_tag,
                            span_columns=True,
                            callback=self._on_row_clicked,
                            user_data=("high_cal", i),
                        )
                        dpg.add_text(str(adim))
                        dpg.add_text("512")
                        dpg.add_text(val_str, tag=val_tag, color=(100, 255, 100) if val is not None else (150, 150, 150))
                self._built_high_steps = list(self.high_cal_steps)
            else:
                for i, adim in enumerate(self.high_cal_steps):
                    val = self.high_cal_data.get(adim)
                    val_str = f"{val:.2f}" if val is not None else "---"
                    val_tag = f"high_val_{self.uuid}_{i}"
                    if dpg.does_item_exist(val_tag):
                        dpg.set_value(val_tag, val_str)
                        dpg.configure_item(val_tag, color=(100, 255, 100) if val is not None else (150, 150, 150))

        # Update live model plots
        self._refresh_plots()

    def _on_low_model_changed(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        val = app_data if app_data is not None else dpg.get_value(self.low_model_combo_tag)
        self.low_model_type = CalibrationModelFit.LABEL_TO_MODEL_TYPE.get(val, "power")
        is_poly = (self.low_model_type == "polynomial")
        if dpg.does_item_exist(self.low_degree_label_tag):
            dpg.configure_item(self.low_degree_label_tag, show=is_poly)
        if dpg.does_item_exist(self.low_degree_input_tag):
            dpg.configure_item(self.low_degree_input_tag, show=is_poly)
        self._refresh_plots()

    def _on_low_degree_changed(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        val = app_data if app_data is not None else dpg.get_value(self.low_degree_input_tag)
        if val is not None:
            self.low_poly_degree = max(2, min(5, int(val)))
            self._refresh_plots()

    def _on_high_model_changed(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        val = app_data if app_data is not None else dpg.get_value(self.high_model_combo_tag)
        self.high_model_type = CalibrationModelFit.LABEL_TO_MODEL_TYPE.get(val, "linear")
        is_poly = (self.high_model_type == "polynomial")
        if dpg.does_item_exist(self.high_degree_label_tag):
            dpg.configure_item(self.high_degree_label_tag, show=is_poly)
        if dpg.does_item_exist(self.high_degree_input_tag):
            dpg.configure_item(self.high_degree_input_tag, show=is_poly)
        self._refresh_plots()

    def _on_high_degree_changed(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        val = app_data if app_data is not None else dpg.get_value(self.high_degree_input_tag)
        if val is not None:
            self.high_poly_degree = max(2, min(5, int(val)))
            self._refresh_plots()

    def _ensure_plot_themes(self) -> None:
        """Ensures fixed themes for plot series so colors don't cycle or change on refresh."""
        self.theme_red_pts = f"theme_red_pts_{self.uuid}"
        if not dpg.does_item_exist(self.theme_red_pts):
            with dpg.theme(tag=self.theme_red_pts):
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, [180, 20, 20, 255], category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, [255, 60, 60, 240], category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_Line, [255, 60, 60, 255], category=dpg.mvThemeCat_Plots)

        self.theme_low_pts = f"theme_low_pts_{self.uuid}"
        if not dpg.does_item_exist(self.theme_low_pts):
            with dpg.theme(tag=self.theme_low_pts):
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, [0, 110, 175, 255], category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, [0, 145, 210, 230], category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_Line, [0, 145, 210, 255], category=dpg.mvThemeCat_Plots)

        self.theme_low_line = f"theme_low_line_{self.uuid}"
        if not dpg.does_item_exist(self.theme_low_line):
            with dpg.theme(tag=self.theme_low_line):
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, [0, 220, 255, 255], category=dpg.mvThemeCat_Plots)

        self.theme_high_pts = f"theme_high_pts_{self.uuid}"
        if not dpg.does_item_exist(self.theme_high_pts):
            with dpg.theme(tag=self.theme_high_pts):
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, [165, 60, 0, 255], category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, [195, 80, 0, 230], category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_Line, [195, 80, 0, 255], category=dpg.mvThemeCat_Plots)

        self.theme_high_line = f"theme_high_line_{self.uuid}"
        if not dpg.does_item_exist(self.theme_high_line):
            with dpg.theme(tag=self.theme_high_line):
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, [255, 120, 0, 255], category=dpg.mvThemeCat_Plots)

    def _refresh_plots(self) -> None:
        """Updates live plots and fitted curves for both Low and High regimes in real-time."""
        self._ensure_plot_themes()

        # 1. Low Regime Plot & Fit
        low_ready = False
        fwd_low = None
        inv_low = None
        pwms = []
        ints = []
        for p in self.low_cal_steps:
            v = self.low_cal_data.get(p)
            if v is not None:
                pwms.append(float(p))
                ints.append(float(v))

        if dpg.does_item_exist(self.low_yaxis_tag):
            children = dpg.get_item_children(self.low_yaxis_tag, 1) or []
            for child in children:
                dpg.delete_item(child)

            # Draw line first (behind points)
            if len(pwms) >= 2:
                low_ready, eq_low, r2_low, fwd_low, inv_low, _ = CalibrationModelFit.fit(
                    pwms, ints, self.low_model_type, self.low_poly_degree
                )
                if low_ready and fwd_low is not None:
                    min_x = max(1.0 if self.low_model_type == "power" else 0.0, float(min(pwms)))
                    max_x = max(float(max(pwms)), 75.0)
                    x_curve = np.linspace(min_x, max_x, 100)
                    y_curve = fwd_low(x_curve)

                    l_tag = dpg.add_line_series(
                        x_curve.tolist(),
                        y_curve.tolist(),
                        label="Fit Curve",
                        parent=self.low_yaxis_tag,
                    )
                    dpg.bind_item_theme(l_tag, self.theme_low_line)

                    if dpg.does_item_exist(self.low_model_text_tag):
                        dpg.set_value(
                            self.low_model_text_tag,
                            f"R² = {r2_low:.4f}   ({len(pwms)} points)",
                        )
                        dpg.configure_item(self.low_model_text_tag, color=(100, 255, 100))
                else:
                    if dpg.does_item_exist(self.low_model_text_tag):
                        dpg.set_value(self.low_model_text_tag, f"R² = --- ({len(pwms)} points)")
                        dpg.configure_item(self.low_model_text_tag, color=(255, 200, 100))
            else:
                if dpg.does_item_exist(self.low_model_text_tag):
                    dpg.set_value(self.low_model_text_tag, f"R² = --- (need >= 2 points)")
                    dpg.configure_item(self.low_model_text_tag, color=(200, 200, 200))

            # Draw points second (rendered in RED ON TOP of the line)
            if pwms:
                s_tag = dpg.add_scatter_series(pwms, ints, label="Measured Points", parent=self.low_yaxis_tag)
                dpg.bind_item_theme(s_tag, self.theme_red_pts)

            if pwms:
                dpg.fit_axis_data(self.low_xaxis_tag)
                dpg.fit_axis_data(self.low_yaxis_tag)

        # 2. High Regime Plot & Fit
        high_ready = False
        fwd_high = None
        inv_high = None
        adims = []
        ints_h = []
        for ad in self.high_cal_steps:
            v = self.high_cal_data.get(ad)
            if v is not None:
                adims.append(float(ad))
                ints_h.append(float(v))

        if dpg.does_item_exist(self.high_yaxis_tag):
            children = dpg.get_item_children(self.high_yaxis_tag, 1) or []
            for child in children:
                dpg.delete_item(child)

            # Draw line first (behind points)
            if len(adims) >= 2:
                high_ready, eq_high, r2_high, fwd_high, inv_high, _ = CalibrationModelFit.fit(
                    adims, ints_h, self.high_model_type, self.high_poly_degree
                )
                if high_ready and fwd_high is not None:
                    min_x = max(1.0 if self.high_model_type == "power" else 0.0, float(min(adims)))
                    max_x = max(float(max(adims)), 512.0)
                    x_curve = np.linspace(min_x, max_x, 100)
                    y_curve = fwd_high(x_curve)

                    l_tag = dpg.add_line_series(
                        x_curve.tolist(),
                        y_curve.tolist(),
                        label="Fit Curve",
                        parent=self.high_yaxis_tag,
                    )
                    dpg.bind_item_theme(l_tag, self.theme_high_line)

                    if dpg.does_item_exist(self.high_model_text_tag):
                        dpg.set_value(
                            self.high_model_text_tag,
                            f"R² = {r2_high:.4f}   ({len(adims)} points)",
                        )
                        dpg.configure_item(self.high_model_text_tag, color=(100, 255, 100))
                else:
                    if dpg.does_item_exist(self.high_model_text_tag):
                        dpg.set_value(self.high_model_text_tag, f"R² = --- ({len(adims)} points)")
                        dpg.configure_item(self.high_model_text_tag, color=(255, 200, 100))
            else:
                if dpg.does_item_exist(self.high_model_text_tag):
                    dpg.set_value(self.high_model_text_tag, f"R² = --- (need >= 2 points)")
                    dpg.configure_item(self.high_model_text_tag, color=(200, 200, 200))

            # Draw points second (rendered in RED ON TOP of the line)
            if adims:
                s_tag = dpg.add_scatter_series(adims, ints_h, label="Measured Points", parent=self.high_yaxis_tag)
                dpg.bind_item_theme(s_tag, self.theme_red_pts)

            if adims:
                dpg.fit_axis_data(self.high_xaxis_tag)
                dpg.fit_axis_data(self.high_yaxis_tag)

        # 3. Combined Plot
        if dpg.does_item_exist(self.comb_yaxis_tag):
            children = dpg.get_item_children(self.comb_yaxis_tag, 1) or []
            for child in children:
                dpg.delete_item(child)

            thresh_val = float(self.threshold)
            all_intensities = list(ints) + list(ints_h)

            # Check if Adjust (visual alignment) is enabled
            adjust_active = False
            if dpg.does_item_exist(self.adjust_checkbox_tag):
                adjust_active = bool(dpg.get_value(self.adjust_checkbox_tag))

            # Low points: (Intensity in µE, PWM in %)
            low_x, low_y_pct = [], []
            for p, v in zip(pwms, ints):
                low_x.append(v)
                low_y_pct.append((p / 512.0) * 100.0)

            # High points: (Intensity in µE, ADIM in %)
            high_x, high_y_pct = [], []
            for ad, v in zip(adims, ints_h):
                high_x.append(v)
                high_y_pct.append((ad / 512.0) * 100.0)

            # 3.1 Visual Adjustment Calculation (global Y-shift applied to ALL High points and the chosen High model)
            is_adjusted = False
            adj_delta = 0.0
            display_high_y = list(high_y_pct)

            if adjust_active and low_ready and high_ready and inv_low is not None and inv_high is not None and high_x:
                try:
                    max_low_x = max(low_x) if low_x else thresh_val
                    overlap_diffs = []

                    for hx, hy in zip(high_x, high_y_pct):
                        if hx <= max_low_x or hx <= thresh_val:
                            # Expected Y% on Low curve
                            y_low_cmd = float(np.clip(inv_low(max(0.01, hx)), 0.0, 512.0))
                            y_low_pct = (y_low_cmd / 512.0) * 100.0
                            overlap_diffs.append(y_low_pct - hy)

                    if overlap_diffs:
                        adj_delta = float(np.mean(overlap_diffs))
                        is_adjusted = True
                    else:
                        # Anchor at threshold if no measured High points fall in the low range
                        y_low_thresh_cmd = float(np.clip(inv_low(max(0.01, thresh_val)), 0.0, 512.0))
                        y_low_thresh_pct = (y_low_thresh_cmd / 512.0) * 100.0
                        y_high_thresh_cmd = float(np.clip(inv_high(max(0.01, thresh_val)), 0.0, 512.0))
                        y_high_thresh_pct = (y_high_thresh_cmd / 512.0) * 100.0
                        adj_delta = y_low_thresh_pct - y_high_thresh_pct
                        is_adjusted = True

                    # Apply global Y-shift to ALL High points
                    display_high_y = [float(np.clip(y + adj_delta, 0.0, 100.0)) for y in high_y_pct]
                except Exception as e:
                    logger.debug(f"Comb adjust shift error: {e}")

            # 3.2 Draw Lines First (behind points)
            if low_ready and inv_low is not None:
                x_low = np.linspace(0.1, thresh_val, 80)
                y_low_pwm = np.clip(inv_low(x_low), 0.0, 512.0)
                y_low_pct = (y_low_pwm / 512.0) * 100.0
                l_low_tag = dpg.add_line_series(
                    x_low.tolist(),
                    y_low_pct.tolist(),
                    label=f"Low {self.low_model_type.title()} Model [<= {int(thresh_val)} µE]",
                    parent=self.comb_yaxis_tag,
                )
                dpg.bind_item_theme(l_low_tag, self.theme_low_line)

            if high_ready and inv_high is not None:
                max_x = max(max(all_intensities) if all_intensities else 2100.0, thresh_val + 200.0)
                x_high = np.linspace(thresh_val, max_x, 80)
                y_high_adim = np.clip(inv_high(x_high), 0.0, 512.0)
                y_high_pct = np.clip(((y_high_adim / 512.0) * 100.0) + (adj_delta if is_adjusted else 0.0), 0.0, 100.0)
                label_high_curve = (
                    f"High {self.high_model_type.title()} Model (Shift Adjusted) [>= {int(thresh_val)} µE]"
                    if is_adjusted
                    else f"High {self.high_model_type.title()} Model [>= {int(thresh_val)} µE]"
                )

                l_high_tag = dpg.add_line_series(
                    x_high.tolist(),
                    y_high_pct.tolist(),
                    label=label_high_curve,
                    parent=self.comb_yaxis_tag,
                )
                dpg.bind_item_theme(l_high_tag, self.theme_high_line)

            # 3.3 Draw Scatter Points Second (rendered ON TOP of the lines)
            if low_x:
                s_low_tag = dpg.add_scatter_series(low_x, low_y_pct, label="Low Points (PWM %)", parent=self.comb_yaxis_tag)
                dpg.bind_item_theme(s_low_tag, self.theme_low_pts)

            if high_x:
                label_high_pts = "High Points (ADIM % - Adjusted)" if is_adjusted else "High Points (ADIM %)"
                s_high_tag = dpg.add_scatter_series(high_x, display_high_y, label=label_high_pts, parent=self.comb_yaxis_tag)
                dpg.bind_item_theme(s_high_tag, self.theme_high_pts)

            if dpg.does_item_exist(self.drag_line_tag):
                dpg.set_value(self.drag_line_tag, float(self.threshold))

            if dpg.does_item_exist(self.comb_model_text_tag):
                adj_str = f" [Shift Adjusted: ΔY = {adj_delta:+.2f}%]" if is_adjusted else ""
                if low_ready and high_ready:
                    dpg.set_value(self.comb_model_text_tag, f"Threshold = {self.threshold} µE{adj_str}")
                    dpg.configure_item(self.comb_model_text_tag, color=(100, 255, 100))
                else:
                    dpg.set_value(self.comb_model_text_tag, f"Threshold = {self.threshold} µE{adj_str}")
                    dpg.configure_item(self.comb_model_text_tag, color=(255, 215, 0))

            if all_intensities:
                dpg.fit_axis_data(self.comb_xaxis_tag)
                dpg.fit_axis_data(self.comb_yaxis_tag)

    def _on_thresh_input_changed(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        val = app_data if app_data is not None else dpg.get_value(self.thresh_input_tag)
        if val is not None:
            try:
                self.threshold = max(1, int(val))
                if dpg.does_item_exist(self.drag_line_tag):
                    dpg.set_value(self.drag_line_tag, float(self.threshold))
                self._refresh_plots()
            except (ValueError, TypeError):
                pass

    def _on_drag_line_changed(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        val = dpg.get_value(self.drag_line_tag) if dpg.does_item_exist(self.drag_line_tag) else app_data
        if val is not None:
            new_thresh = max(1, int(round(float(val))))
            self.threshold = new_thresh
            if dpg.does_item_exist(self.thresh_input_tag):
                dpg.set_value(self.thresh_input_tag, new_thresh)
            self._refresh_plots()

    def _on_prev_step(self, *args: Any, **kwargs: Any) -> None:
        if self.current_mode == "high_cal":
            if self.current_step_idx > 0:
                self._set_active_step("high_cal", self.current_step_idx - 1)
            else:
                # Transition back to the end of Low Mode
                self._set_active_step("low_cal", len(self.low_cal_steps) - 1)
                if dpg.does_item_exist(self.tab_bar_tag):
                    dpg.set_value(self.tab_bar_tag, self.tab_low_tag)
        else:
            if self.current_step_idx > 0:
                self._set_active_step("low_cal", self.current_step_idx - 1)

    def _on_next_step(self, *args: Any, **kwargs: Any) -> None:
        if self.current_mode == "low_cal":
            if self.current_step_idx < len(self.low_cal_steps) - 1:
                self._set_active_step("low_cal", self.current_step_idx + 1)
            else:
                # Transition seamlessly to the start of High Mode
                self._set_active_step("high_cal", 0)
                if dpg.does_item_exist(self.tab_bar_tag):
                    dpg.set_value(self.tab_bar_tag, self.tab_high_tag)
        else:
            if self.current_step_idx < len(self.high_cal_steps) - 1:
                self._set_active_step("high_cal", self.current_step_idx + 1)

    def _on_record_and_next(self, *args: Any, **kwargs: Any) -> None:
        """Record the measured intensity for the current step and advance."""
        raw_val = dpg.get_value(self.measured_ue_tag) if dpg.does_item_exist(self.measured_ue_tag) else None
        try:
            measured = float(raw_val) if raw_val is not None else 0.0
        except (ValueError, TypeError):
            measured = 0.0

        if self.current_mode == "low_cal":
            if 0 <= self.current_step_idx < len(self.low_cal_steps):
                pwm = self.low_cal_steps[self.current_step_idx]
                self.low_cal_data[pwm] = measured
                logger.info(f"Recorded Low Cal: PWM {pwm} -> {measured} µE")

            if self.current_step_idx < len(self.low_cal_steps) - 1:
                next_mode = "low_cal"
                next_idx = self.current_step_idx + 1
            else:
                # At the end of Low Mode, automatically advance to High Mode!
                logger.info("Completed all Low Mode steps! Transitioning to High Mode.")
                next_mode = "high_cal"
                next_idx = 0
                if dpg.does_item_exist(self.status_msg_tag):
                    dpg.set_value(self.status_msg_tag, "Low Mode finished! Switched to High Mode.")
                    dpg.configure_item(self.status_msg_tag, color=(100, 255, 100))
        else:
            if 0 <= self.current_step_idx < len(self.high_cal_steps):
                adim = self.high_cal_steps[self.current_step_idx]
                self.high_cal_data[adim] = measured
                logger.info(f"Recorded High Cal: ADIM {adim} -> {measured} µE")

            if self.current_step_idx < len(self.high_cal_steps) - 1:
                next_mode = "high_cal"
                next_idx = self.current_step_idx + 1
            else:
                logger.info("Completed all High Mode steps!")
                next_mode = "high_cal"
                next_idx = self.current_step_idx
                if dpg.does_item_exist(self.status_msg_tag):
                    dpg.set_value(self.status_msg_tag, "High Mode finished! You can save.")
                    dpg.configure_item(self.status_msg_tag, color=(100, 255, 100))

        self.current_mode = next_mode
        self.current_step_idx = next_idx
        self._refresh_tables()
        self._set_active_step(next_mode, next_idx)

    def _on_test_press(self, *args: Any, **kwargs: Any) -> None:
        """Called when the TEST button is clicked/pressed down."""
        self._is_testing = True
        step_val = self._get_active_step_value()

        if self.current_mode == "low_cal":
            # PWM variation: ADIM is fixed at 512
            cmd = f"act 512 {step_val}\n"
        else:
            # ADIM variation: PWM is fixed at 512
            cmd = f"act {step_val} 512\n"

        if self.serial.connected:
            self.serial.send_command(cmd)
            logger.debug(f"Calibration TEST firing: {cmd.strip()}")
        else:
            logger.warning("Serial not connected! Cannot send TEST light command.")

        if dpg.does_item_exist(self.status_msg_tag):
            dpg.set_value(self.status_msg_tag, "Status: LIGHT ON (Hold...)")
            dpg.configure_item(self.status_msg_tag, color=(255, 200, 0))

    def _on_test_release(self, *args: Any, **kwargs: Any) -> None:
        """Called when the mouse is released anywhere."""
        if not self._is_testing:
            return

        self._is_testing = False
        # Turn off actinic LED immediately
        cmd = "act 0 0\n"
        if self.serial.connected:
            self.serial.send_command(cmd)
            logger.debug("Calibration TEST released: act 0 0")

        if dpg.does_item_exist(self.status_msg_tag):
            dpg.set_value(self.status_msg_tag, "Status: OFF")
            dpg.configure_item(self.status_msg_tag, color=(160, 160, 160))

    def _on_save_calibration(self, *args: Any, **kwargs: Any) -> None:
        """Saves current calibration data to a JSON file via file_explorer."""
        # Compile lists
        low_pwm = []
        low_intensity = []
        for pwm in self.low_cal_steps:
            val = self.low_cal_data.get(pwm)
            if val is not None:
                low_pwm.append(pwm)
                low_intensity.append(float(val))

        high_adim = []
        high_intensity = []
        for adim in self.high_cal_steps:
            val = self.high_cal_data.get(adim)
            if val is not None:
                high_adim.append(adim)
                high_intensity.append(float(val))

        if len(low_pwm) < 2 or len(high_adim) < 2:
            logger.warning("Need at least 2 points for both Low and High calibration to save.")
            if dpg.does_item_exist(self.status_msg_tag):
                dpg.set_value(self.status_msg_tag, "Error: Need >= 2 points for Low and High!")
                dpg.configure_item(self.status_msg_tag, color=(255, 80, 80))
            return

        thresh_val = int(dpg.get_value(self.thresh_input_tag)) if dpg.does_item_exist(self.thresh_input_tag) else self.threshold

        low_model_dict: Dict[str, Any] = {"type": self.low_model_type}
        if self.low_model_type == "polynomial":
            low_model_dict["degree"] = self.low_poly_degree

        high_model_dict: Dict[str, Any] = {"type": self.high_model_type}
        if self.high_model_type == "polynomial":
            high_model_dict["degree"] = self.high_poly_degree

        data_dict = {
            "act_calibration": {
                "threshold": thresh_val,
                "low_cal": {
                    "model": low_model_dict,
                    "pwm": low_pwm,
                    "intensity": low_intensity,
                },
                "high_cal": {
                    "model": high_model_dict,
                    "adim": high_adim,
                    "intensity": high_intensity,
                },
            }
        }

        default_dir = os.path.dirname(__file__)

        def on_save_selected(file_path: str, ext: str, *a: Any) -> None:
            if not file_path:
                return
            if not file_path.endswith(".json"):
                file_path += ".json"

            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data_dict, f, indent=4)
                logger.success(f"Successfully saved calibration to {file_path}")

                if dpg.does_item_exist(self.status_msg_tag):
                    dpg.set_value(self.status_msg_tag, f"Saved: {os.path.basename(file_path)}")
                    dpg.configure_item(self.status_msg_tag, color=(100, 255, 100))

                if self.on_calibration_saved:
                    self.on_calibration_saved(file_path)

            except Exception as e:
                logger.error(f"Failed to write calibration JSON: {e}")
                if dpg.does_item_exist(self.status_msg_tag):
                    dpg.set_value(self.status_msg_tag, f"Save failed: {e}")
                    dpg.configure_item(self.status_msg_tag, color=(255, 80, 80))

        file_explorer.save_file(
            default_path=default_dir,
            default_name="fluo_calibration.json",
            extensions=[("JSON files", "*.json"), ("All files", "*.*")],
            callback=on_save_selected,
        )

    def _apply_calibration_dict(self, data: dict, file_path: Optional[str] = None) -> bool:
        act_cal = data.get("act_calibration")
        if not act_cal:
            logger.warning(f"No 'act_calibration' section in data ({file_path})")
            return False

        self.threshold = int(act_cal.get("threshold", 150))
        if file_path:
            self.loaded_file_path = file_path

        # Low cal
        low = act_cal.get("low_cal", {})
        low_m_obj = low.get("model")
        if isinstance(low_m_obj, dict):
            self.low_model_type = low_m_obj.get("type", "power")
            self.low_poly_degree = int(low_m_obj.get("degree", 2))
        elif isinstance(low_m_obj, str):
            self.low_model_type = low_m_obj
            self.low_poly_degree = 2
        else:
            self.low_model_type = "power"
            self.low_poly_degree = 2

        pwms = low.get("pwm", [])
        ints_low = low.get("intensity", [])
        if pwms:
            self.low_cal_steps = [int(p) for p in pwms]
            self.low_cal_data = {}
            for i, p in enumerate(self.low_cal_steps):
                self.low_cal_data[p] = float(ints_low[i]) if i < len(ints_low) else None

        # High cal
        high = act_cal.get("high_cal", {})
        high_m_obj = high.get("model")
        if isinstance(high_m_obj, dict):
            self.high_model_type = high_m_obj.get("type", "linear")
            self.high_poly_degree = int(high_m_obj.get("degree", 2))
        elif isinstance(high_m_obj, str):
            self.high_model_type = high_m_obj
            self.high_poly_degree = 2
        else:
            self.high_model_type = "linear"
            self.high_poly_degree = 2

        adims = high.get("adim", [])
        ints_high = high.get("intensity", [])
        if adims:
            self.high_cal_steps = [int(a) for a in adims]
            self.high_cal_data = {}
            for i, a in enumerate(self.high_cal_steps):
                self.high_cal_data[a] = float(ints_high[i]) if i < len(ints_high) else None

        self.current_step_idx = 0
        return True

    def load_from_dict(self, data: dict, file_path: Optional[str] = None) -> bool:
        """Loads calibration from dictionary and updates all live UI items."""
        if not self._apply_calibration_dict(data, file_path=file_path):
            return False

        if dpg.does_item_exist(self.thresh_input_tag):
            dpg.set_value(self.thresh_input_tag, self.threshold)
        if dpg.does_item_exist(self.drag_line_tag):
            dpg.set_value(self.drag_line_tag, float(self.threshold))
        if dpg.does_item_exist(self.comb_model_text_tag):
            dpg.set_value(self.comb_model_text_tag, f"Threshold: {self.threshold} µE")

        if dpg.does_item_exist(self.low_model_combo_tag):
            dpg.set_value(
                self.low_model_combo_tag,
                CalibrationModelFit.MODEL_TYPE_TO_LABEL.get(self.low_model_type, "Power Law (a * x^b)"),
            )
        if dpg.does_item_exist(self.low_degree_input_tag):
            dpg.set_value(self.low_degree_input_tag, self.low_poly_degree)
            dpg.configure_item(self.low_degree_input_tag, show=(self.low_model_type == "polynomial"))
        if dpg.does_item_exist(self.low_degree_label_tag):
            dpg.configure_item(self.low_degree_label_tag, show=(self.low_model_type == "polynomial"))

        if dpg.does_item_exist(self.high_model_combo_tag):
            dpg.set_value(
                self.high_model_combo_tag,
                CalibrationModelFit.MODEL_TYPE_TO_LABEL.get(self.high_model_type, "Linear (a * x + b)"),
            )
        if dpg.does_item_exist(self.high_degree_input_tag):
            dpg.set_value(self.high_degree_input_tag, self.high_poly_degree)
            dpg.configure_item(self.high_degree_input_tag, show=(self.high_model_type == "polynomial"))
        if dpg.does_item_exist(self.high_degree_label_tag):
            dpg.configure_item(self.high_degree_label_tag, show=(self.high_model_type == "polynomial"))

        if dpg.does_item_exist(self.win_tag):
            self._refresh_tables()
            self._set_active_step(self.current_mode, self.current_step_idx)

        if file_path:
            logger.success(f"Calibration imported from {file_path}")
            if dpg.does_item_exist(self.status_msg_tag):
                dpg.set_value(self.status_msg_tag, f"Loaded: {os.path.basename(file_path)}")
                dpg.configure_item(self.status_msg_tag, color=(100, 255, 100))

        return True

    def load_from_file(self, file_path: str) -> bool:
        """Loads calibration JSON file and updates all UI elements."""
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"Calibration file does not exist: {file_path}")
            return False
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self.load_from_dict(data, file_path=file_path)
        except Exception as e:
            logger.error(f"Failed to load calibration JSON from {file_path}: {e}")
            if dpg.does_item_exist(self.status_msg_tag):
                dpg.set_value(self.status_msg_tag, f"Load failed: {e}")
                dpg.configure_item(self.status_msg_tag, color=(255, 80, 80))
            return False

    def _on_load_calibration(self, *args: Any, **kwargs: Any) -> None:
        """Loads existing calibration data from JSON via file_explorer."""
        default_dir = os.path.dirname(__file__)

        def on_load_selected(file_path: str, ext: str, *a: Any) -> None:
            if not file_path or not os.path.exists(file_path):
                return
            if self.load_from_file(file_path):
                if self.on_calibration_saved:
                    self.on_calibration_saved(file_path)

        file_explorer.select_file(
            default_path=default_dir,
            extensions=[("JSON files", "*.json"), ("All files", "*.*")],
            callback=on_load_selected,
        )

    def _on_reset_defaults(self, *args: Any, **kwargs: Any) -> None:
        """Resets steps and cleared data to factory defaults."""
        self.low_cal_steps = list(self.DEFAULT_LOW_PWM)
        self.low_cal_data = {pwm: None for pwm in self.low_cal_steps}

        self.high_cal_steps = list(self.DEFAULT_HIGH_ADIM)
        self.high_cal_data = {adim: None for adim in self.high_cal_steps}

        self.threshold = 150
        self.loaded_file_path = None
        if dpg.does_item_exist(self.thresh_input_tag):
            dpg.set_value(self.thresh_input_tag, 150)
        if dpg.does_item_exist(self.drag_line_tag):
            dpg.set_value(self.drag_line_tag, 150.0)
        if dpg.does_item_exist(self.comb_model_text_tag):
            dpg.set_value(self.comb_model_text_tag, "Threshold: 150 µE")

        self.current_mode = "low_cal"
        self.current_step_idx = 0
        self._refresh_tables()
        self._set_active_step("low_cal", 0)

        if dpg.does_item_exist(self.tab_bar_tag):
            dpg.set_value(self.tab_bar_tag, self.tab_low_tag)

        if dpg.does_item_exist(self.status_msg_tag):
            dpg.set_value(self.status_msg_tag, "Reset to default steps.")
            dpg.configure_item(self.status_msg_tag, color=(160, 160, 160))

    def is_ready(self) -> bool:
        """Indicate whether the dialog window is ready."""
        return True

    def on_close(self) -> None:
        """Turns off light and hides window."""
        self._on_test_release()
        if dpg.does_item_exist(self.win_tag):
            dpg.hide_item(self.win_tag)

    def close(self) -> None:
        """Closes the calibration dialog."""
        self.destroy()

    def destroy(self) -> None:
        """Turns off light, cleans up handlers, themes, and destroys window."""
        self._on_test_release()
        self._built_low_steps = []
        self._built_high_steps = []
        for tag in [
            self.test_handler_tag,
            self.global_handler_tag,
            self.drag_line_tag,
            self.win_tag,
            f"theme_red_pts_{self.uuid}",
            f"theme_low_pts_{self.uuid}",
            f"theme_low_line_{self.uuid}",
            f"theme_high_pts_{self.uuid}",
            f"theme_high_line_{self.uuid}",
        ]:
            if dpg.does_item_exist(tag):
                try:
                    dpg.delete_item(tag)
                except Exception:
                    pass


FluoCalibration_win = FluoCalibrationWin
__all__ = ["FluoCalibrationWin", "FluoCalibration_win", "CalibrationModelFit"]
