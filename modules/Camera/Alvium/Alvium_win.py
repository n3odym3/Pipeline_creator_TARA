from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional, Union
import threading
import queue
import os
import json
import numpy as np
import cv2
import dearpygui.dearpygui as dpg
from loguru import logger

from core.window_base import WindowBase
from core.input_output_types import IOTypes
from config.display_scaling import display_scaling
from core.config_manager import config
from .Alvium import Alvium_cam


class Alvium_win(WindowBase):
    """
    Window module for controlling Alvium cameras.
    """

    description: str = "Control Allied Vision Alvium cameras. Supports exposure, gain, white balance, and trigger modes."

    init_btn_tag: str
    trigger_tag: str
    start_btn_tag: str
    exposure_tag: str
    gain_tag: str
    wb_r_tag: str
    wb_b_tag: str
    binning_tag: str
    format_tag: str
    roi_w_tag: str
    roi_h_tag: str
    sat_enabled_tag: str
    sat_factor_tag: str
    gamma_enabled_tag: str
    gamma_val_tag: str
    ffc_enabled_tag: str
    ffc_status_tag: str

    cam_running: bool
    cam_thread: Optional[threading.Thread]
    alvium_cam: Alvium_cam

    ffc_enabled: bool
    ffc_matrix_mono: Optional[np.ndarray]
    ffc_matrix_rgb: Optional[np.ndarray]
    ffc_gain_mono: Optional[np.ndarray]
    ffc_gain_rgb: Optional[np.ndarray]
    _capture_ffc_next: bool

    sat_enabled: bool
    sat_factor: float
    gamma_enabled: bool
    gamma_val: float
    _gamma_lut_8: Optional[np.ndarray]
    _gamma_lut_16: Optional[np.ndarray]

    def __init__(
        self,
        label: str = "Alvium Camera",
        win_width: int = display_scaling.scale(300),
        win_height: int = display_scaling.scale(400),
        pos: Tuple[int, int] = (10, 10),
        uuid: Optional[str] = None,
        outputs: Optional[Dict[str, Any]] = None,
        visible: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            label=label,
            pos=pos,
            win_width=win_width,
            win_height=win_height,
            uuid=uuid,
            outputs=outputs,
            visible=visible,
        )

        self.init_btn_tag = f"alvium_init_btn_{self.UUID}"
        self.trigger_tag = f"alvium_trigger_{self.UUID}"
        self.start_btn_tag = f"alvium_start_btn_{self.UUID}"
        self.exposure_tag = f"alvium_exposure_{self.UUID}"
        self.gain_tag = f"alvium_gain_{self.UUID}"
        self.wb_r_tag = f"alvium_wb_r_{self.UUID}"
        self.wb_b_tag = f"alvium_wb_b_{self.UUID}"
        self.binning_tag = f"alvium_binning_{self.UUID}"
        self.format_tag = f"alvium_format_{self.UUID}"
        self.roi_w_tag = f"alvium_roi_w_{self.UUID}"
        self.roi_h_tag = f"alvium_roi_h_{self.UUID}"
        self.sat_enabled_tag = f"alvium_sat_enabled_{self.UUID}"
        self.sat_factor_tag = f"alvium_sat_factor_{self.UUID}"
        self.gamma_enabled_tag = f"alvium_gamma_enabled_{self.UUID}"
        self.gamma_val_tag = f"alvium_gamma_val_{self.UUID}"
        self.ffc_enabled_tag = f"alvium_ffc_enabled_{self.UUID}"
        self.ffc_status_tag = f"alvium_ffc_status_{self.UUID}"

        self.cam_running = False
        self.cam_thread = None
        self.alvium_cam = Alvium_cam()

        self.ffc_enabled = False
        self.ffc_matrix_mono = None
        self.ffc_matrix_rgb = None
        self.ffc_gain_mono = None
        self.ffc_gain_rgb = None
        self._capture_ffc_next = False

        self._persistent_fields = ["label"]
        self.sat_enabled = False
        self.sat_factor = 1.0
        self.gamma_enabled = False
        self.gamma_val = 1.0
        self._gamma_lut_8 = None
        self._gamma_lut_16 = None
        self.accepted_input_types = [IOTypes.CMD_DICT]
        self.outputs = {
            "Frame": IOTypes.FRAME,
        }
        self.connections = {k: [] for k in self.outputs}

        with dpg.window(
            label=self.label,
            width=self.win_width,
            height=self.win_height,
            pos=self.pos,
            tag=self.winID,
            show=self.visible,
        ):
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Initialize",
                    tag=self.init_btn_tag,
                    callback=self._init_camera,
                )
                dpg.add_button(
                    label="Start",
                    tag=self.start_btn_tag,
                    callback=self._start_capture,
                )
                dpg.add_button(label="Stop", callback=self._stop_capture)

            dpg.add_separator()

            with dpg.group(horizontal=True):
                dpg.add_combo(
                    items=["Video", "Trigger"],
                    default_value="Video",
                    tag=self.trigger_tag,
                    callback=self._set_trigger,
                    width=display_scaling.scale(100),
                )
                dpg.add_button(
                    label="Trigger",
                    callback=lambda *args: self.alvium_cam.trigger(),
                )

            with dpg.group(horizontal=True):
                dpg.add_input_float(
                    label="Expos. (ms)",
                    default_value=1.0,
                    min_value=0.01,
                    max_value=1000,
                    format="%.2f",
                    step=-1,
                    tag=self.exposure_tag,
                    callback=self._set_exposure,
                    on_enter=True,
                    width=display_scaling.scale(100),
                )
                dpg.add_button(label="Auto Once", callback=self._auto_exposure)

            dpg.add_input_int(
                label="Gain (dB)",
                default_value=1,
                min_value=0,
                max_value=24,
                step=-1,
                tag=self.gain_tag,
                callback=self._set_gain,
                on_enter=True,
                width=display_scaling.scale(100),
            )

            dpg.add_combo(
                label="Format",
                items=[
                    "RGB8 (ISP)",
                    "Mono12",
                    "BayerGR12 (Raw)",
                    "BayerGR12 (RGB)",
                    "BayerGR12 (Red)",
                    "BayerGR12 (Green)",
                    "BayerGR12 (Blue)",
                ],
                default_value="RGB8 (ISP)",
                tag=self.format_tag,
                callback=self._set_format,
                width=display_scaling.scale(150),
            )

            dpg.add_combo(
                label="Binning",
                items=["1", "2", "4", "8"],
                default_value="1",
                tag=self.binning_tag,
                callback=self._set_binning,
                width=display_scaling.scale(100),
            )

            dpg.add_separator()
            dpg.add_text("White Balance")
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Auto Once", callback=self._auto_white_balance
                )

            dpg.add_drag_float(
                label="WB Red",
                default_value=1.0,
                min_value=0.1,
                max_value=10.0,
                tag=self.wb_r_tag,
                callback=self._set_manual_wb,
                width=display_scaling.scale(70),
                speed=0.01,
            )
            dpg.add_drag_float(
                label="WB Blue",
                default_value=1.0,
                min_value=0.1,
                max_value=10.0,
                tag=self.wb_b_tag,
                callback=self._set_manual_wb,
                width=display_scaling.scale(70),
                speed=0.01,
            )

            dpg.add_separator()
            dpg.add_text("ROI / Resolution")
            with dpg.group(horizontal=True):
                dpg.add_input_int(
                    label="W",
                    tag=self.roi_w_tag,
                    step=-1,
                    width=display_scaling.scale(70),
                )
                dpg.add_input_int(
                    label="H",
                    tag=self.roi_h_tag,
                    step=-1,
                    width=display_scaling.scale(70),
                )
                dpg.add_button(label="Apply", callback=self._apply_roi)
                dpg.add_button(label="Max", callback=self._max_roi)

            dpg.add_separator()
            with dpg.collapsing_header(
                label="Image Processing", default_open=True
            ):
                dpg.add_checkbox(
                    label="Enable Saturation",
                    tag=self.sat_enabled_tag,
                    callback=lambda s, v: setattr(self, "sat_enabled", v),
                )
                dpg.add_slider_float(
                    label="Saturation Factor",
                    tag=self.sat_factor_tag,
                    default_value=1.0,
                    min_value=0.0,
                    max_value=3.0,
                    callback=lambda s, v: setattr(self, "sat_factor", v),
                    width=display_scaling.scale(150),
                )

                dpg.add_separator()

                dpg.add_checkbox(
                    label="Enable Gamma",
                    tag=self.gamma_enabled_tag,
                    callback=lambda s, v: setattr(self, "gamma_enabled", v),
                )
                dpg.add_slider_float(
                    label="Gamma Value",
                    tag=self.gamma_val_tag,
                    default_value=1.0,
                    min_value=0.1,
                    max_value=3.0,
                    callback=self._update_gamma,
                    width=display_scaling.scale(150),
                )

                dpg.add_separator()
                dpg.add_text("Flat Field Correction (FFC)")
                dpg.add_checkbox(
                    label="Enable FFC",
                    tag=self.ffc_enabled_tag,
                    callback=lambda s, v: setattr(self, "ffc_enabled", v),
                )
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Take Ref Image",
                        callback=self._request_ffc_capture,
                    )
                    dpg.add_text(
                        "None",
                        tag=self.ffc_status_tag,
                        color=(200, 100, 100),
                    )

            dpg.add_separator()
            dpg.add_button(
                label="Save Settings (Config)",
                callback=self.save_settings,
                width=-1,
            )

        self.autosize_window()
        self.update_permission()

    def update_permission(self) -> None:
        """
        Adjust module permissions and UI visibility based on the application mode.
        """
        from core.app_state import app_state
        is_user = app_state.mode == "user"

        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def _init_camera(
        self,
        sender: Optional[str] = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
    ) -> None:
        with dpg.theme() as theme_init:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(
                    dpg.mvThemeCol_Button,
                    (255, 165, 0),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonHovered,
                    (255, 180, 50),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonActive,
                    (255, 140, 0),
                    category=dpg.mvThemeCat_Core,
                )
            dpg.bind_item_theme(self.init_btn_tag, theme_init)

        def init_task() -> None:
            if self.alvium_cam.init():
                self.load_settings()
                if self.alvium_cam.had_error:
                    with dpg.theme() as theme_fail:
                        with dpg.theme_component(dpg.mvAll):
                            dpg.add_theme_color(
                                dpg.mvThemeCol_Button,
                                (150, 0, 0),
                                category=dpg.mvThemeCat_Core,
                            )
                            dpg.add_theme_color(
                                dpg.mvThemeCol_ButtonHovered,
                                (200, 0, 0),
                                category=dpg.mvThemeCat_Core,
                            )
                            dpg.add_theme_color(
                                dpg.mvThemeCol_ButtonActive,
                                (255, 0, 0),
                                category=dpg.mvThemeCat_Core,
                            )
                    dpg.bind_item_theme(self.init_btn_tag, theme_fail)
                    logger.error("Alvium camera configuration failed during settings application")
                    return

                with dpg.theme() as theme_success:
                    with dpg.theme_component(dpg.mvAll):
                        dpg.add_theme_color(
                            dpg.mvThemeCol_Button,
                            (0, 100, 0),
                            category=dpg.mvThemeCat_Core,
                        )
                        dpg.add_theme_color(
                            dpg.mvThemeCol_ButtonHovered,
                            (0, 150, 0),
                            category=dpg.mvThemeCat_Core,
                        )
                        dpg.add_theme_color(
                            dpg.mvThemeCol_ButtonActive,
                            (0, 200, 0),
                            category=dpg.mvThemeCat_Core,
                        )
                dpg.bind_item_theme(self.init_btn_tag, theme_success)

                logger.success("Alvium camera initialized")
                self._start_capture()
            else:
                with dpg.theme() as theme_fail:
                    with dpg.theme_component(dpg.mvAll):
                        dpg.add_theme_color(
                            dpg.mvThemeCol_Button,
                            (150, 0, 0),
                            category=dpg.mvThemeCat_Core,
                        )
                        dpg.add_theme_color(
                            dpg.mvThemeCol_ButtonHovered,
                            (200, 0, 0),
                            category=dpg.mvThemeCat_Core,
                        )
                        dpg.add_theme_color(
                            dpg.mvThemeCol_ButtonActive,
                            (255, 0, 0),
                            category=dpg.mvThemeCat_Core,
                        )
                dpg.bind_item_theme(self.init_btn_tag, theme_fail)
                logger.error("Alvium camera initialization failed")

        threading.Thread(target=init_task, daemon=True).start()

    def _start_capture(
        self,
        sender: Optional[str] = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
    ) -> None:
        if not self.alvium_cam.defined or self.alvium_cam.had_error:
            logger.error("Cannot start capture: Alvium camera is in an error state or is not initialized.")
            return

        if self.cam_thread and self.cam_thread.is_alive():
            return

        self.cam_running = True

        with dpg.theme() as theme_active:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(
                    dpg.mvThemeCol_Button,
                    (0, 100, 0),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonHovered,
                    (0, 150, 0),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonActive,
                    (0, 200, 0),
                    category=dpg.mvThemeCat_Core,
                )
            dpg.bind_item_theme(self.start_btn_tag, theme_active)

        self.alvium_cam.start_streaming()

        def capture_task() -> None:
            while self.cam_running and not self.alvium_cam.had_error:
                try:
                    frame = self.alvium_cam.frame_queue.get(timeout=0.01)
                    self._trigger_outputs(frame)
                except queue.Empty:
                    pass
            self.alvium_cam.stop_streaming()
            self.cam_running = False
            dpg.bind_item_theme(self.start_btn_tag, 0)

        self.cam_thread = threading.Thread(target=capture_task, daemon=True)
        self.cam_thread.start()

    def _stop_capture(
        self,
        sender: Optional[str] = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
    ) -> None:
        if not self.cam_running:
            return
        self.cam_running = False
        if self.cam_thread:
            self.cam_thread.join(timeout=1.0)
        dpg.bind_item_theme(self.start_btn_tag, 0)

    def _apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        """
        Rotate frame based on camera configuration.
        """
        angle = config.get("Cameras", {}).get("Alvium", {}).get("rotation", 0)
        if angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _trigger_outputs(self, frame: np.ndarray) -> None:
        frame = self._apply_rotation(frame)

        if self._capture_ffc_next:
            self._capture_ffc_next = False
            self._process_ffc_reference(frame)

        if self.ffc_enabled:
            try:
                is_color = len(frame.shape) == 3
                gain = self.ffc_gain_rgb if is_color else self.ffc_gain_mono

                if gain is not None:
                    dtype = frame.dtype
                    if frame.shape[:2] != gain.shape[:2]:
                        gain = cv2.resize(gain, (frame.shape[1], frame.shape[0]))

                    if is_color:
                        frame = frame.astype(np.float32) * np.expand_dims(gain, axis=2)
                    else:
                        frame = frame.astype(np.float32) * gain

                    is_12bit = "12" in self.alvium_cam.processing_mode
                    max_clip = 4095.0 if is_12bit else (65535.0 if dtype == np.uint16 else 255.0)
                    frame = np.clip(frame, 0, max_clip).astype(dtype)
            except Exception as e:
                logger.error(f"FFC application failed: {e}")

        if self.sat_enabled and len(frame.shape) == 3 and frame.shape[2] >= 3:
            try:
                is_16bit = frame.dtype == np.uint16
                is_12bit = "12" in self.alvium_cam.processing_mode
                divisor = 4095.0 if is_12bit else (65535.0 if is_16bit else 255.0)

                hsv = cv2.cvtColor(frame.astype(np.float32) / divisor, cv2.COLOR_BGR2HSV)
                hsv[:, :, 1] *= self.sat_factor
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1.0)

                frame_float = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                frame = (frame_float * divisor).clip(0, divisor).astype(frame.dtype)
            except Exception as e:
                logger.error(f"Saturation adjustment failed: {e}")

        if self.gamma_enabled:
            try:
                if frame.dtype == np.uint8:
                    if self._gamma_lut_8 is None:
                        self._update_gamma(None, self.gamma_val)
                    if self._gamma_lut_8 is not None:
                        frame = cv2.LUT(frame, self._gamma_lut_8)
                else:
                    if self._gamma_lut_16 is None:
                        self._update_gamma(None, self.gamma_val)
                    if self._gamma_lut_16 is not None:
                        frame = self._gamma_lut_16[frame]
            except Exception as e:
                logger.error(f"Gamma correction failed: {e}")

        data = (frame, "Alvium")
        for output_key, connected_modules in self.connections.items():
            for module in connected_modules:
                try:
                    module.input_cb(data)
                except Exception as e:
                    logger.error(f"Error sending frame to {module}: {e}")

    def _update_gamma(self, sender: Any, app_data: Any, user_data: Any = None) -> None:
        self.gamma_val = app_data
        invGamma = 1.0 / max(0.1, app_data)

        self._gamma_lut_8 = np.array([((i / 255.0) ** invGamma) * 255
                                   for i in np.arange(0, 256)]).astype("uint8")

        is_12bit = "12" in self.alvium_cam.processing_mode
        divisor = 4095.0 if is_12bit else 65535.0

        self._gamma_lut_16 = np.array([((i / divisor) ** invGamma) * divisor
                                    for i in np.arange(0, 65536)])
        self._gamma_lut_16 = np.clip(self._gamma_lut_16, 0, divisor).astype(np.uint16)

    def _request_ffc_capture(self, *args: Any) -> None:
        if not self.cam_running:
            logger.warning("Camera must be running to take FFC reference")
            return
        self._capture_ffc_next = True
        logger.info("FFC reference capture requested - point camera at uniform source")

    def _process_ffc_reference(self, frame: np.ndarray) -> None:
        try:
            is_color = len(frame.shape) == 3

            if is_color:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()

            gray_f = gray.astype(np.float32)
            gray_f = cv2.GaussianBlur(gray_f, (31, 31), 0)

            peak = np.max(gray_f)
            if peak == 0:
                peak = 1.0

            matrix = gray_f / peak
            matrix[matrix < 0.05] = 0.05
            gain = 1.0 / matrix

            if is_color:
                self.ffc_matrix_rgb = matrix
                self.ffc_gain_rgb = gain
                f_path = os.path.join(os.path.dirname(__file__), "alvium_ffc_rgb.npy")
            else:
                self.ffc_matrix_mono = matrix
                self.ffc_gain_mono = gain
                f_path = os.path.join(os.path.dirname(__file__), "alvium_ffc_mono.npy")

            np.save(f_path, matrix)

            if dpg.does_item_exist(self.ffc_status_tag):
                dpg.set_value(self.ffc_status_tag, f"OK ({'RGB' if is_color else 'Mono'})")
                dpg.configure_item(self.ffc_status_tag, color=(100, 255, 100))
            logger.success(f"FFC {'RGB' if is_color else 'Mono'} reference captured as Grayscale matrix")

        except Exception as e:
            logger.error(f"FFC processing failed: {e}")
            if dpg.does_item_exist(self.ffc_status_tag):
                dpg.set_value(self.ffc_status_tag, "Error")
                dpg.configure_item(self.ffc_status_tag, color=(255, 100, 100))

    def _set_trigger(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any) -> None:
        if self.alvium_cam.defined:
            self.alvium_cam.set_trigger_mode(app_data)

    def _set_exposure(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any) -> None:
        if self.alvium_cam.defined:
            exposure_us = app_data * 1000
            self.alvium_cam.set_exposure(exposure_us)

    def _set_gain(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any) -> None:
        if self.alvium_cam.defined:
            self.alvium_cam.set_gain(app_data)

    def _set_manual_wb(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any) -> None:
        if self.alvium_cam.defined:
            r = dpg.get_value(self.wb_r_tag)
            b = dpg.get_value(self.wb_b_tag)
            self.alvium_cam.set_manual_white_balance(r, b)

    def _set_format(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any) -> None:
        if self.alvium_cam.defined:
            self.alvium_cam.set_format(app_data)

    def _set_binning(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any) -> None:
        if self.alvium_cam.defined:
            self.alvium_cam.set_binning(app_data)
            self._max_roi()

    def _apply_roi(self, *args: Any) -> None:
        if self.alvium_cam.defined:
            w = dpg.get_value(self.roi_w_tag)
            h = dpg.get_value(self.roi_h_tag)
            self.alvium_cam.set_roi(w, h)

    def _max_roi(self, *args: Any) -> None:
        if self.alvium_cam.defined:
            w, h = self.alvium_cam.get_max_roi()
            dpg.set_value(self.roi_w_tag, w)
            dpg.set_value(self.roi_h_tag, h)
            self.alvium_cam.set_roi(w, h)

    def _auto_exposure(self, *args: Any) -> None:
        self.alvium_cam.auto_exposure()
        def pull_values() -> None:
            import time
            time.sleep(1.0)
            if self.alvium_cam.defined:
                expos_us = self.alvium_cam.get_exposure()
                dpg.set_value(self.exposure_tag, int(round(expos_us / 1000.0)))
        threading.Thread(target=pull_values, daemon=True).start()

    def _auto_white_balance(self, *args: Any) -> None:
        self.alvium_cam.auto_white_balance()
        def pull_values() -> None:
            import time
            time.sleep(1.0)
            if self.alvium_cam.defined:
                r, b = self.alvium_cam.get_white_balance()
                dpg.set_value(self.wb_r_tag, r)
                dpg.set_value(self.wb_b_tag, b)
        threading.Thread(target=pull_values, daemon=True).start()

    def save_settings(self, *args: Any) -> None:
        try:
            f = os.path.join(os.path.dirname(__file__), "alvium_settings.json")
            data = {
                "exposure": dpg.get_value(self.exposure_tag),
                "gain": dpg.get_value(self.gain_tag),
                "format": dpg.get_value(self.format_tag),
                "binning": dpg.get_value(self.binning_tag),
                "roi_w": dpg.get_value(self.roi_w_tag),
                "roi_h": dpg.get_value(self.roi_h_tag),
                "wb_r": dpg.get_value(self.wb_r_tag),
                "wb_b": dpg.get_value(self.wb_b_tag),
                "sat_enabled": dpg.get_value(self.sat_enabled_tag),
                "sat_factor": dpg.get_value(self.sat_factor_tag),
                "gamma_enabled": dpg.get_value(self.gamma_enabled_tag),
                "gamma_val": dpg.get_value(self.gamma_val_tag),
                "ffc_enabled": dpg.get_value(self.ffc_enabled_tag),
            }
            with open(f, "w") as fh:
                json.dump(data, fh)
            logger.success("Alvium settings saved to alvium_settings.json")
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")

    def load_settings(self, *args: Any) -> None:
        try:
            f = os.path.join(os.path.dirname(__file__), "alvium_settings.json")
            if os.path.exists(f):
                with open(f, "r") as fh:
                    data = json.load(fh)
                dpg.set_value(self.exposure_tag, int(data.get("exposure", 2)))
                self._set_exposure(None, int(data.get("exposure", 2)))
                dpg.set_value(self.gain_tag, data.get("gain", 1.0))
                self._set_gain(None, data.get("gain", 1.0))

                fmt = str(data.get("format", "RGB8 (ISP)"))
                dpg.set_value(self.format_tag, fmt)
                self.alvium_cam.set_format(fmt)

                binning = str(data.get("binning", "1"))
                dpg.set_value(self.binning_tag, binning)
                self.alvium_cam.set_binning(binning)

                rw, rh = self.alvium_cam.get_roi()
                dpg.set_value(self.roi_w_tag, data.get("roi_w", rw))
                dpg.set_value(self.roi_h_tag, data.get("roi_h", rh))
                self.alvium_cam.set_roi(dpg.get_value(self.roi_w_tag), dpg.get_value(self.roi_h_tag))

                dpg.set_value(self.wb_r_tag, data.get("wb_r", 1.0))
                dpg.set_value(self.wb_b_tag, data.get("wb_b", 1.0))
                self._set_manual_wb(None, None)

                self.sat_enabled = data.get("sat_enabled", False)
                dpg.set_value(self.sat_enabled_tag, self.sat_enabled)
                self.sat_factor = data.get("sat_factor", 1.0)
                dpg.set_value(self.sat_factor_tag, self.sat_factor)

                self.gamma_enabled = data.get("gamma_enabled", False)
                dpg.set_value(self.gamma_enabled_tag, self.gamma_enabled)
                self.gamma_val = data.get("gamma_val", 1.0)
                dpg.set_value(self.gamma_val_tag, self.gamma_val)
                self._update_gamma(None, self.gamma_val)

                self.ffc_enabled = data.get("ffc_enabled", False)
                dpg.set_value(self.ffc_enabled_tag, self.ffc_enabled)

                p_mono = os.path.join(os.path.dirname(__file__), "alvium_ffc_mono.npy")
                p_rgb = os.path.join(os.path.dirname(__file__), "alvium_ffc_rgb.npy")

                status_parts = []
                if os.path.exists(p_mono):
                    try:
                        self.ffc_matrix_mono = np.load(p_mono)
                        self.ffc_gain_mono = 1.0 / self.ffc_matrix_mono
                        status_parts.append("Mono")
                    except Exception:
                        pass
                if os.path.exists(p_rgb):
                    try:
                        self.ffc_matrix_rgb = np.load(p_rgb)
                        self.ffc_gain_rgb = 1.0 / self.ffc_matrix_rgb
                        status_parts.append("RGB")
                    except Exception:
                        pass

                if status_parts and dpg.does_item_exist(self.ffc_status_tag):
                    dpg.set_value(self.ffc_status_tag, f"OK ({'+'.join(status_parts)})")
                    dpg.configure_item(self.ffc_status_tag, color=(100, 255, 100))

                logger.info("Restored Alvium settings from config")
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")

    def is_ready(self) -> bool:
        return True

    def input_cb(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
        if data is None and "data" in kwargs:
            data = kwargs["data"]

        if isinstance(data, dict):
            cmd = data
        elif kwargs.get("data_type") == IOTypes.CMD_DICT:
            cmd = kwargs.get("data")
        else:
            cmd = None

        if cmd and isinstance(cmd, dict):
            if "trigger" in cmd or "trigger_cam" in cmd:
                self.alvium_cam.trigger()
            if "init" in cmd and cmd["init"]:
                self._init_camera()
            if "trigger_mode" in cmd:
                mode = cmd["trigger_mode"]
                self._set_trigger(None, mode)
                dpg.set_value(self.trigger_tag, mode)
            if "exposure" in cmd:
                val_ms = int(cmd["exposure"])
                self._set_exposure(None, val_ms)
                dpg.set_value(self.exposure_tag, val_ms)
            if "gain" in cmd:
                val = float(cmd["gain"])
                self._set_gain(None, val)
                dpg.set_value(self.gain_tag, val)

    def close(self) -> None:
        self._stop_capture()
        self.alvium_cam.close()
        super().close()


EXPORTED_CLASS = Alvium_win
EXPORTED_NAME = "Alvium Camera"

