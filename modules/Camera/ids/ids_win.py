from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional, Union
import threading
import time
import numpy as np
import cv2
import dearpygui.dearpygui as dpg
from loguru import logger

from core.window_base import WindowBase
from core.input_output_types import IOTypes
from core.config_manager import config
from .ids import IDS

class IDS_win(WindowBase):
    """
    Window module for controlling IDS industrial cameras.
    """

    description: str = "Control IDS industrial cameras. Supports exposure, gain, trigger modes, and 12-bit frame output."

    init_btn_tag: str
    _tag_fm_status: str
    _suppress_emission: bool
    gain_tag: str
    exposure_tag: str
    pixel_clock_tag: str
    shutter_tag: str
    trigger_tag: str
    start_btn_tag: str
    width_tag: str
    height_tag: str

    cam_running: bool
    cam_thread: Optional[threading.Thread]
    ids_cam: IDS

    dark_ref_frame: Optional[np.ndarray]
    collecting_dark_ref: bool
    dark_ref_n_images: int
    dark_ref_accumulator: Optional[np.ndarray]
    dark_ref_counter: int
    correct_dark: bool
    dark_threshold: int

    width: Optional[int]
    height: Optional[int]

    def __init__(
        self,
        label: str = "IDS Camera",
        win_width: int = 350,
        win_height: int = 300,
        pos: Tuple[int, int] = (10, 10),
        uuid: Optional[str] = None,
        outputs: Optional[Dict[str, Any]] = None,
        visible: bool = True,
        width: Optional[int] = None,
        height: Optional[int] = None,
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

        self.init_btn_tag = f"ids_init_btn_{self.UUID}"
        self._tag_fm_status = f"fsel_fm_status_{self.UUID}"
        self._suppress_emission = False
        self.gain_tag = f"ids_gain_{self.UUID}"
        self.exposure_tag = f"ids_exposure_{self.UUID}"
        self.pixel_clock_tag = f"ids_pixelclock_{self.UUID}"
        self.shutter_tag = f"ids_shutter_{self.UUID}"
        self.trigger_tag = f"ids_trigger_{self.UUID}"
        self.start_btn_tag = f"ids_start_btn_{self.UUID}"
        self.width_tag = f"ids_width_{self.UUID}"
        self.height_tag = f"ids_height_{self.UUID}"

        self.cam_running = False
        self.cam_thread = None
        self.ids_cam = IDS()

        self.dark_ref_frame = None
        self.collecting_dark_ref = False
        self.dark_ref_n_images = 10
        self.dark_ref_accumulator = None
        self.dark_ref_counter = 0
        self.correct_dark = True
        self.dark_threshold = 10

        self._persistent_fields = ["label", "width", "height"]
        self.width = width
        self.height = height
        self.accepted_input_types = [IOTypes.CMD_DICT]
        self.outputs = {
            "12bit frame": IOTypes.FRAME12,
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
                    width=150,
                )
                dpg.add_button(
                    label="Trigger",
                    callback=lambda *args: self.ids_cam.trigger(),
                )

            dpg.add_drag_int(
                label="Expos. (µs)",
                default_value=2000,
                min_value=1,
                max_value=5000,
                tag=self.exposure_tag,
                callback=self._set_exposure,
                width=100,
            )

            with dpg.collapsing_header(label="Advanced"):
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Ref frame", callback=self._start_dark_ref_capture
                    )
                    dpg.add_button(label="Clear", callback=self._clear_dark_ref)

                dpg.add_input_int(
                    label="N Ref rep",
                    default_value=self.dark_ref_n_images,
                    width=100,
                    step=-1,
                    callback=lambda s, a, *args: setattr(
                        self, "dark_ref_n_images", a
                    ),
                )

                dpg.add_checkbox(
                    label="Correct dark",
                    default_value=self.correct_dark,
                    callback=lambda s, a, *args: setattr(
                        self, "correct_dark", a
                    ),
                )

                dpg.add_input_int(
                    label="Dark threshold",
                    default_value=self.dark_threshold,
                    width=100,
                    step=-1,
                    callback=lambda s, a, *args: setattr(
                        self, "dark_threshold", a
                    ),
                )

                dpg.add_slider_int(
                    label="Gain",
                    default_value=0,
                    min_value=0,
                    max_value=100,
                    tag=self.gain_tag,
                    callback=self._set_gain,
                    width=200,
                )

                dpg.add_separator()
                dpg.add_text("AOI / Resolution")
                with dpg.group(horizontal=True):
                    dpg.add_input_int(
                        label="Width",
                        default_value=self.width if self.width is not None else 1280,
                        tag=self.width_tag,
                        width=100,
                        step=0,
                        callback=lambda s, a: setattr(self, "width", a),
                    )
                    dpg.add_input_int(
                        label="Height",
                        default_value=self.height if self.height is not None else 1024,
                        tag=self.height_tag,
                        width=100,
                        step=0,
                        callback=lambda s, a: setattr(self, "height", a),
                    )
                dpg.add_button(
                    label="Update Resolution", callback=self._apply_aoi
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
            try:
                if self.ids_cam.init():
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

                    logger.success("IDS camera initialized")

                    self.width = int(self.ids_cam.width)
                    self.height = int(self.ids_cam.height)

                    if dpg.does_item_exist(self.width_tag):
                        dpg.set_value(self.width_tag, self.width)
                    if dpg.does_item_exist(self.height_tag):
                        dpg.set_value(self.height_tag, self.height)

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
                    logger.error("IDS camera initialization failed")
            except Exception as e:
                with dpg.theme() as theme_error:
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
                dpg.bind_item_theme(self.init_btn_tag, theme_error)
                logger.error(f"IDS init error: {e}")

        threading.Thread(target=init_task, daemon=True).start()

    def _apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        """
        Rotate frame based on camera configuration.
        """
        angle = config.get("Cameras", {}).get("IDS", {}).get("rotation", 0)
        if angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _start_capture(
        self,
        sender: Optional[str] = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
    ) -> None:
        if not self.ids_cam.defined:
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

        def capture_task() -> None:
            while self.cam_running:
                if self.ids_cam.check_event(timeout_ms=100):
                    frame = self.ids_cam.get_image()
                    if frame is not None:
                        frame = self._apply_rotation(frame)

                        is_collecting = self.collecting_dark_ref

                        if is_collecting:
                            if self.dark_ref_accumulator is None:
                                self.dark_ref_accumulator = frame.astype(np.float32)
                            else:
                                self.dark_ref_accumulator += frame.astype(np.float32)

                            self.dark_ref_counter += 1

                            if self.dark_ref_counter >= self.dark_ref_n_images:
                                self.dark_ref_frame = self.dark_ref_accumulator / self.dark_ref_counter
                                self.collecting_dark_ref = False
                                logger.success(f"Dark reference captured from {self.dark_ref_counter} frames")

                        ref_frame = self.dark_ref_frame
                        if self.correct_dark and ref_frame is not None:
                            corrected = frame.astype(np.float32) - ref_frame

                            mask = corrected < self.dark_threshold
                            if np.any(mask):
                                dynamic_offset = np.mean(corrected[mask])
                                corrected -= dynamic_offset

                            corrected = np.clip(corrected, 0, 65535)
                            frame = corrected.astype(np.uint16)

                        if not is_collecting and not self._suppress_emission:
                            self._trigger_outputs(frame)

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

    def _trigger_outputs(self, frame: np.ndarray) -> None:
        data = (frame, "IDS")
        for output_key, connected_modules in self.connections.items():
            for module in connected_modules:
                try:
                    module.input_cb(data)
                except Exception as e:
                    logger.error(f"Error sending frame to {module}: {e}")

    def _set_pixel_clock(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any) -> None:
        if self.ids_cam.defined:
            self.ids_cam.set_pixel_clock(app_data)

    def _set_gain(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any) -> None:
        if self.ids_cam.defined:
            self.ids_cam.set_gain(app_data)

    def _set_exposure(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any) -> None:
        if self.ids_cam.defined:
            self.ids_cam.set_exposure(app_data)

    def _set_shutter(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any) -> None:
        if self.ids_cam.defined:
            self.ids_cam.set_shutter(app_data)

    def _set_trigger(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any) -> None:
        if self.ids_cam.defined:
            self.ids_cam.set_trigger(app_data)

    def _apply_aoi(self, sender: Optional[str] = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        if self.ids_cam.defined:
            was_running = self.cam_running
            if was_running:
                self._stop_capture()

            w = dpg.get_value(self.width_tag)
            h = dpg.get_value(self.height_tag)

            w = (w // 2) * 2
            h = (h // 2) * 2

            res = self.ids_cam.set_aoi(w, h)
            if isinstance(res, tuple):
                self.width, self.height = res
                dpg.set_value(self.width_tag, self.width)
                dpg.set_value(self.height_tag, self.height)

            if was_running:
                self._start_capture()

            logger.info(f"IDS Resolution updated to {self.width}x{self.height}")

    def _start_dark_ref_capture(self, sender: Optional[str] = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        self.dark_ref_accumulator = None
        self.dark_ref_counter = 0
        self.collecting_dark_ref = True
        self._suppress_emission = True
        logger.info(f"Starting dark reference capture ({self.dark_ref_n_images} frames)")

        original_mode = dpg.get_value(self.trigger_tag)

        if original_mode != "Trigger":
            self._set_trigger(None, "Trigger")
            dpg.set_value(self.trigger_tag, "Trigger")

        def capture_sequence() -> None:
            time.sleep(0.1)
            expos_us = dpg.get_value(self.exposure_tag)
            sleep_time = (expos_us / 1000000.0) + 0.1

            for _ in range(self.dark_ref_n_images):
                if not self.collecting_dark_ref:
                    break
                self.ids_cam.trigger()
                time.sleep(sleep_time)

            timeout = 2.0
            start = time.time()
            while self.collecting_dark_ref and time.time() - start < timeout:
                time.sleep(0.1)

            self.collecting_dark_ref = False

            if original_mode != "Trigger":
                self._set_trigger(None, original_mode)
                dpg.set_value(self.trigger_tag, original_mode)

            time.sleep(0.5)
            self._suppress_emission = False

        threading.Thread(target=capture_sequence, daemon=True).start()

    def _clear_dark_ref(self, sender: Optional[str] = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        self.dark_ref_frame = None
        self.collecting_dark_ref = False
        logger.info("Dark reference cleared")

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
            if "trigger_cam" in cmd:
                if "exposure" in cmd:
                    def trigger_seq() -> None:
                        old_exposure = dpg.get_value(self.exposure_tag)
                        new_exposure = int(cmd["exposure"])
                        self._set_exposure(None, new_exposure)
                        dpg.set_value(self.exposure_tag, new_exposure)
                        time.sleep(0.1)
                        self.ids_cam.trigger()
                        time.sleep((new_exposure / 1000000.0) + 0.1)
                        self._set_exposure(None, old_exposure)
                        dpg.set_value(self.exposure_tag, old_exposure)

                    threading.Thread(target=trigger_seq, daemon=True).start()
                else:
                    self.ids_cam.trigger()

            if "trigger" in cmd:
                self.ids_cam.trigger()

            if "gain" in cmd:
                val = int(cmd["gain"])
                self._set_gain(None, val)
                dpg.set_value(self.gain_tag, val)

            if "exposure" in cmd and "trigger_cam" not in cmd and "start_dark_ref" not in cmd:
                val = int(cmd["exposure"])
                self._set_exposure(None, val)
                dpg.set_value(self.exposure_tag, val)

            if "start_dark_ref" in cmd:
                if "n_frames" in cmd:
                    self.dark_ref_n_images = int(cmd["n_frames"])
                self._start_dark_ref_capture()

            if "trigger_mode" in cmd:
                mode = cmd["trigger_mode"]
                self._set_trigger(None, mode)
                dpg.set_value(self.trigger_tag, mode)

            if "init" in cmd and cmd["init"]:
                self._init_camera()

            if "width" in cmd and "height" in cmd:
                self.width = int(cmd["width"])
                self.height = int(cmd["height"])
                dpg.set_value(self.width_tag, self.width)
                dpg.set_value(self.height_tag, self.height)
                self._apply_aoi()

    def close(self) -> None:
        self._stop_capture()
        if self.ids_cam.defined:
            try:
                self.ids_cam.close()
            except Exception:
                pass
        super().close()


EXPORTED_CLASS = IDS_win
EXPORTED_NAME = "IDS Camera"
