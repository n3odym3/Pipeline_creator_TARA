from __future__ import annotations

import threading
import uuid as uuid_mod
from pathlib import Path
from typing import Any, Optional

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import torch
from loguru import logger
from segment_anything import SamPredictor, sam_model_registry

from config.display_scaling import display_scaling
from core.input_output_types import IOTypes
from core.window_base import WindowBase

# --- SAM CONFIGURATION ---
# Default weight file and architecture
SAM_CHECKPOINT_NAME = "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
# -------------------------


class SamInteractiveWin(WindowBase):
    """
    Interactive Segment Anything Model (SAM) window module.

    Provides point-based interactive segmentation with positive/negative prompts,
    GPU/CPU acceleration, pre-processing filters, and multi-format mask/overlay outputs.
    """

    label: str = "SAM Interactive"
    win_width: int = display_scaling.scale(600)
    win_height: int = display_scaling.scale(650)
    description: str = (
        "Interactive Segment Anything Model (SAM) with point prompting, pre-processing, and multi-format mask outputs."
    )

    accepted_input_types: list[type] = [
        IOTypes.FRAME,
        IOTypes.FRAME12,
        IOTypes.FRAME16,
        IOTypes.CMD_DICT,
    ]
    outputs: dict[str, type] = {
        "Overlay": IOTypes.FRAME,
        "Mask": IOTypes.FRAME,
        "ROI": IOTypes.ROI_SAMPLE,
        "Result": IOTypes.FRAME_MASK_PAIR,
    }

    process_enabled: bool = False
    saturation: float = 1.0
    gamma: float = 1.0

    persistent_fields: list[str] = [
        "label",
        "process_enabled",
        "saturation",
        "gamma",
    ]

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the SAM Interactive window module.

        Args:
            **kwargs: Extra persistent configuration fields forwarded to WindowBase.
        """
        super().__init__(**kwargs)

        raw_proc = getattr(self, "process_enabled", None)
        self.process_enabled: bool = bool(raw_proc) if raw_proc is not None else False

        raw_sat = getattr(self, "saturation", None)
        self.saturation: float = float(raw_sat) if raw_sat is not None else 1.0

        raw_gam = getattr(self, "gamma", None)
        self.gamma: float = float(raw_gam) if raw_gam is not None else 1.0

        self.enabled: bool = True

        self._frame: Optional[np.ndarray] = None
        self._rgb_frame: Optional[np.ndarray] = None
        self._raw_bgr_frame: Optional[np.ndarray] = None
        self._current_mask: Optional[np.ndarray] = None

        self._lock = threading.Lock()

        # Concurrency & queue tracking
        self._is_encoding: bool = False
        self._pending_image: Optional[np.ndarray] = None
        self._predictor_frame_shape: Optional[tuple[int, int]] = None
        self._is_predicting: bool = False
        self._pending_predict: bool = False

        # UI Tags for image processing controls
        uid = self.UUID
        self._tag_proc_enable: str = f"sam_proc_enable_{uid}"
        self._tag_proc_sat: str = f"sam_proc_sat_{uid}"
        self._tag_proc_gamma: str = f"sam_proc_gamma_{uid}"

        # Model initialization (loaded once at instantiation)
        self.predictor: Optional[SamPredictor] = None
        self._load_sam_model()

        self.points: list[list[int]] = []
        self.labels: list[int] = []

        # UI Tags
        self._tag_plot: str = f"sam_plot_{uid}"
        self._tag_texture: str = f"sam_texture_{uid}"
        self._tag_texture_reg: str = f"sam_texture_reg_{uid}"
        dpg.add_texture_registry(tag=self._tag_texture_reg, show=False)

        self._tag_status: str = f"sam_status_{uid}"
        self._tag_reset_btn: str = f"sam_reset_{uid}"

        self.texture_width: int = 100
        self.texture_height: int = 100
        self._tex_buffer: Optional[np.ndarray] = None
        self._sat_lut_val: Optional[float] = None
        self._sat_lut: Optional[np.ndarray] = None
        self._gamma_lut_val: Optional[float] = None
        self._gamma_lut: Optional[np.ndarray] = None
        self._roi_uuid: str = str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, f"{uid}_sam_roi"))
        self.mouse_x: int = 0
        self.mouse_y: int = 0
        self.is_hovering: bool = False
        self._tag_handler_reg: str = f"sam_handlers_{uid}"

        self._build_ui()
        self.update_permission()

    def update_permission(self) -> None:
        """
        Adjust module permissions and UI visibility based on app_state.mode.
        Disable closing in user mode.
        """
        from core.app_state import app_state

        is_user = app_state.mode == "user"

        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def is_ready(self) -> bool:
        """Return True if model predictor is loaded and ready."""
        return self.predictor is not None

    def on_close(self) -> None:
        """Clean up registered DearPyGui handlers and textures upon closing."""
        if dpg.does_item_exist(self._tag_handler_reg):
            try:
                dpg.delete_item(self._tag_handler_reg)
            except Exception:
                pass
        if dpg.does_item_exist(self._tag_texture_reg):
            try:
                dpg.delete_item(self._tag_texture_reg)
            except Exception:
                pass
        super().on_close()

    def close(self) -> None:
        """Clean up and unregister the window from the registry."""
        self.on_close()
        super().close()

    def destroy(self) -> None:
        """Destroy module resources."""
        self.close()

    def _load_sam_model(self) -> None:
        """Loads the SAM model when the module is created."""
        try:
            ckpt_path = Path(__file__).parent / SAM_CHECKPOINT_NAME
            if not ckpt_path.exists():
                logger.error(f"Weight file not found: {ckpt_path}")
                return

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Initializing SAM ({SAM_MODEL_TYPE}) on {device}...")

            # Instantiate model and load weights with strict=False to support CAT models
            sam = sam_model_registry[SAM_MODEL_TYPE]()
            state_dict = torch.load(str(ckpt_path), map_location=device)

            msg = sam.load_state_dict(state_dict, strict=False)
            if msg.unexpected_keys:
                logger.warning(
                    f"SAM: Some layers ({len(msg.unexpected_keys)}) were ignored (specific CAT layers)."
                )

            sam.to(device=device)
            sam.eval()
            self.predictor = SamPredictor(sam)
            logger.success("SAM loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading SAM: {e}")

    def _build_ui(self) -> None:
        """Construct DearPyGui interface widgets, pre-processing controls, and image viewer."""
        scale = display_scaling.scale

        with self.window():
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Reset Points",
                    tag=self._tag_reset_btn,
                    callback=self.reset_points,
                )
                dpg.add_text("Ready", tag=self._tag_status, color=(100, 200, 100))

            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="Enable Proc.",
                    tag=self._tag_proc_enable,
                    default_value=self.process_enabled,
                    callback=self._on_proc_settings_change,
                )
                dpg.add_text("Sat:")
                dpg.add_drag_float(
                    tag=self._tag_proc_sat,
                    default_value=self.saturation,
                    min_value=0.0,
                    max_value=5.0,
                    speed=0.05,
                    width=scale(80),
                    callback=self._on_proc_settings_change,
                )
                dpg.add_text("Gamma:")
                dpg.add_drag_float(
                    tag=self._tag_proc_gamma,
                    default_value=self.gamma,
                    min_value=0.1,
                    max_value=5.0,
                    speed=0.05,
                    width=scale(80),
                    callback=self._on_proc_settings_change,
                )

            with dpg.group(horizontal=True):
                dpg.add_text("CTRL+Left Click: Include", color=(150, 255, 150))
                dpg.add_text(" CTRL+Right Click: Exclude", color=(255, 150, 150))
            dpg.add_separator()

            self._init_viewer(800, 600)

            with dpg.handler_registry(tag=self._tag_handler_reg):
                dpg.add_mouse_move_handler(callback=self.on_mouse_move)
                dpg.add_mouse_click_handler(callback=self.on_mouse_click)

    def _init_viewer(self, width: int, height: int) -> None:
        """Thread-safe initialisation of viewer texture and plot."""
        from core.automation_manager import automation_manager

        if threading.current_thread() is threading.main_thread():
            self._init_viewer_internal(width, height)
        else:
            automation_manager.post_to_main_thread(
                self._init_viewer_internal, width, height, wait=True
            )

    def _init_viewer_internal(self, width: int, height: int) -> None:
        """Create or recreate the raw texture and plot axes for image rendering."""
        if dpg.does_item_exist(self._tag_texture):
            dpg.delete_item(self._tag_texture)
        if dpg.does_item_exist(self._tag_plot):
            dpg.delete_item(self._tag_plot)

        self.texture_width, self.texture_height = width, height
        self._tex_buffer = np.zeros(height * width * 3, dtype=np.float32)

        dpg.add_raw_texture(
            width,
            height,
            default_value=self._tex_buffer,
            tag=self._tag_texture,
            format=dpg.mvFormat_Float_rgb,
            parent=self._tag_texture_reg,
        )

        plot_tag = dpg.add_plot(
            tag=self._tag_plot,
            no_menus=True,
            no_title=True,
            width=-1,
            height=-1,
            parent=self.winID,
        )
        xaxis = dpg.add_plot_axis(
            dpg.mvXAxis,
            no_gridlines=True,
            no_tick_labels=True,
            no_tick_marks=True,
            parent=plot_tag,
        )
        yaxis = dpg.add_plot_axis(
            dpg.mvYAxis,
            no_gridlines=True,
            no_tick_labels=True,
            no_tick_marks=True,
            parent=plot_tag,
        )
        dpg.add_image_series(self._tag_texture, [0, 0], [width, height], parent=yaxis)
        dpg.fit_axis_data(xaxis)
        dpg.fit_axis_data(yaxis)

    def convert_to_texture(self, frame: np.ndarray, is_rgb: bool = False) -> np.ndarray:
        """Convert BGR or RGB numpy image to normalized float32 RGB for DearPyGui raw textures."""
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        rgb_frame = frame if is_rgb else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        flat = rgb_frame.ravel()
        if self._tex_buffer is None or self._tex_buffer.size != flat.size:
            self._tex_buffer = np.empty(flat.size, dtype=np.float32)
        np.multiply(flat, 1.0 / 255.0, out=self._tex_buffer, dtype=np.float32)
        return self._tex_buffer

    def _get_processed_rgb(self) -> Optional[np.ndarray]:
        """Apply saturation and gamma adjustments to the raw BGR frame and return RGB."""
        if self._raw_bgr_frame is None:
            return None

        bgr = self._raw_bgr_frame
        if self.process_enabled:
            # Saturation adjustment via cached uint8 LUT
            if abs(self.saturation - 1.0) > 1e-4:
                if self._sat_lut is None or self._sat_lut_val != self.saturation:
                    self._sat_lut_val = self.saturation
                    self._sat_lut = np.clip(
                        np.arange(256, dtype=np.float32) * self.saturation, 0, 255
                    ).astype(np.uint8)
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                hsv[:, :, 1] = cv2.LUT(hsv[:, :, 1], self._sat_lut)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Gamma adjustment via cached uint8 LUT
            if abs(self.gamma - 1.0) > 1e-4:
                if self._gamma_lut is None or self._gamma_lut_val != self.gamma:
                    self._gamma_lut_val = self.gamma
                    inv_gamma = 1.0 / max(0.01, self.gamma)
                    self._gamma_lut = (
                        ((np.arange(256, dtype=np.float32) / 255.0) ** inv_gamma) * 255.0
                    ).astype(np.uint8)
                bgr = cv2.LUT(bgr, self._gamma_lut)

        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _on_proc_settings_change(self, *args: Any, **kwargs: Any) -> None:
        """Callback triggered when pre-processing toggles or sliders change."""
        with self._lock:
            if dpg.does_item_exist(self._tag_proc_enable):
                self.process_enabled = bool(dpg.get_value(self._tag_proc_enable))
            if dpg.does_item_exist(self._tag_proc_sat):
                self.saturation = float(dpg.get_value(self._tag_proc_sat))
            if dpg.does_item_exist(self._tag_proc_gamma):
                self.gamma = float(dpg.get_value(self._tag_proc_gamma))

            if self._raw_bgr_frame is not None:
                self._rgb_frame = self._get_processed_rgb()

        if self._raw_bgr_frame is not None:
            if getattr(self, "enabled", True):
                self._trigger_set_image()
            else:
                self._update_display()

    def input_cb(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Handle incoming frame data or command dictionaries from upstream modules.
        """
        if data is None and "data" in kwargs:
            data = kwargs["data"]
        elif data is None and args:
            data = args[0]

        if isinstance(data, dict):
            if data.get("clear", False) or data.get("reset_points", False):
                self.reset_points()
            if "enable" in data or "enabled" in data:
                was_enabled = getattr(self, "enabled", True)
                self.enabled = bool(data.get("enable", data.get("enabled", True)))
                if (
                    self.enabled
                    and not was_enabled
                    and getattr(self, "_rgb_frame", None) is not None
                ):
                    self._trigger_set_image()
            return

        frame = data
        if isinstance(frame, tuple) and len(frame) > 0 and isinstance(frame[0], np.ndarray):
            frame = frame[0]

        if isinstance(frame, np.ndarray):
            h, w = frame.shape[:2]
            if h != self.texture_height or w != self.texture_width:
                self._init_viewer(w, h)

            with self._lock:
                self._frame = frame.copy()
                if len(frame.shape) == 2:
                    frame_8b = (
                        cv2.convertScaleAbs(frame, alpha=255.0 / max(1, int(frame.max())))
                        if frame.dtype != np.uint8
                        else frame
                    )
                    self._raw_bgr_frame = cv2.cvtColor(frame_8b, cv2.COLOR_GRAY2BGR)
                else:
                    self._raw_bgr_frame = (
                        frame.copy()
                        if frame.dtype == np.uint8
                        else cv2.convertScaleAbs(frame, alpha=255.0 / max(1, int(frame.max())))
                    )

                self._rgb_frame = self._get_processed_rgb()

            self.reset_points()
            if getattr(self, "enabled", True):
                self._trigger_set_image()
            else:
                self._update_display()

    def _trigger_set_image(self) -> None:
        """Enqueue the current frame for asynchronous SAM image encoding."""
        with self._lock:
            if self._rgb_frame is None:
                return
            self._pending_image = self._rgb_frame.copy()
            if self._is_encoding:
                return
            self._is_encoding = True

        threading.Thread(target=self._encoding_worker_loop, daemon=True).start()

    def _encoding_worker_loop(self) -> None:
        """Worker loop encoding frames in the SAM predictor background thread."""
        while True:
            with self._lock:
                img = self._pending_image
                self._pending_image = None
                if img is None:
                    self._is_encoding = False
                    break

            try:
                dpg.set_value(self._tag_status, "Encoding image...")
                dpg.configure_item(self._tag_status, color=(255, 255, 100))

                if self.predictor is not None:
                    with torch.inference_mode():
                        self.predictor.set_image(img)
                    with self._lock:
                        self._predictor_frame_shape = img.shape[:2]

                has_points = False
                with self._lock:
                    if self.points:
                        has_points = True

                if has_points:
                    self._trigger_predict()
                else:
                    dpg.set_value(self._tag_status, "Ready")
                    dpg.configure_item(self._tag_status, color=(100, 255, 100))
                    self._update_display()
            except Exception as e:
                logger.error(f"SAM error: {e}")
                dpg.set_value(self._tag_status, f"Error: {e}")

    def reset_points(self, *args: Any, **kwargs: Any) -> None:
        """Clear all prompt points, labels, and mask results."""
        with self._lock:
            self.points, self.labels, self._current_mask = [], [], None
        self._update_display()
        if not getattr(self, "_is_encoding", False):
            dpg.set_value(self._tag_status, "Ready")
            dpg.configure_item(self._tag_status, color=(100, 255, 100))

    def on_mouse_move(self, sender: Any = None, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Track mouse coordinates within the interactive plot."""
        try:
            if dpg.does_item_exist(self._tag_plot) and dpg.is_item_hovered(self._tag_plot):
                self.is_hovering = True
                mpos = dpg.get_plot_mouse_pos()
                self.mouse_x, self.mouse_y = int(mpos[0]), int(self.texture_height - mpos[1])
            else:
                self.is_hovering = False
        except Exception:
            self.is_hovering = False

    def on_mouse_click(self, sender: Any = None, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Handle CTRL+Click prompt point additions (Left=Foreground, Right=Background)."""
        if not getattr(self, "enabled", True):
            return
        if not self.is_hovering or not dpg.is_key_down(dpg.mvKey_LControl):
            return
        if (
            self.mouse_x < 0
            or self.mouse_x >= self.texture_width
            or self.mouse_y < 0
            or self.mouse_y >= self.texture_height
        ):
            return

        with self._lock:
            self.points.append([self.mouse_x, self.mouse_y])
            self.labels.append(1 if dpg.is_mouse_button_down(0) else 0)

        self._trigger_predict()

    def _trigger_predict(self) -> None:
        """Queue and trigger asynchronous mask prediction."""
        with self._lock:
            if not self.points:
                return
            if self._is_encoding:
                dpg.set_value(self._tag_status, "Encoding image... will predict once ready")
                return
            if (
                self._rgb_frame is not None
                and self._predictor_frame_shape != self._rgb_frame.shape[:2]
            ):
                dpg.set_value(self._tag_status, "Syncing image...")
                self._trigger_set_image()
                return

            if self._is_predicting:
                self._pending_predict = True
                return
            self._is_predicting = True

        dpg.set_value(self._tag_status, "Predicting...")
        threading.Thread(target=self._predict_worker_loop, daemon=True).start()

    def _predict_worker_loop(self) -> None:
        """Background thread executing SAM predictor inference."""
        while True:
            with self._lock:
                self._pending_predict = False

            self._predict_worker_internal()

            with self._lock:
                if not self._pending_predict:
                    self._is_predicting = False
                    break

    def _predict_worker_internal(self) -> None:
        """Internal execution of predictor inference, mask selection, and dispatch."""
        try:
            with self._lock:
                if self.predictor is None or not self.points or self._rgb_frame is None:
                    return
                if self._predictor_frame_shape != self._rgb_frame.shape[:2]:
                    logger.warning("SAM: Predict called but predictor not synced with frame shape.")
                    return
                pts, lbls = np.array(self.points), np.array(self.labels)

            with torch.inference_mode():
                masks, scores, _ = self.predictor.predict(
                    point_coords=pts, point_labels=lbls, multimask_output=True
                )
            with self._lock:
                self._current_mask = masks[np.argmax(scores)]

            self._update_display()
            self._export_outputs()
            dpg.set_value(self._tag_status, "Ready. Mask updated.")
            dpg.configure_item(self._tag_status, color=(100, 255, 100))
        except Exception as e:
            logger.error(f"SAM predict failed: {e}")
            dpg.set_value(self._tag_status, f"Error: {e}")

    def _update_display(self) -> None:
        """Render frame, prompt markers, and semi-transparent mask overlay."""
        with self._lock:
            if self._rgb_frame is None:
                return
            display_img = self._rgb_frame.copy()
            mask, pts, lbls = self._current_mask, self.points.copy(), self.labels.copy()

        if mask is not None:
            if mask.shape[:2] != display_img.shape[:2]:
                logger.warning(
                    f"SAM: Mask shape {mask.shape[:2]} does not match display image shape {display_img.shape[:2]}. Discarding mask."
                )
                with self._lock:
                    self._current_mask = None
                mask = None
            else:
                masked = display_img[mask]
                masked >>= 1
                masked[:, 0] += 127
                display_img[mask] = masked

        for (x, y), lbl in zip(pts, lbls):
            color = (0, 255, 0) if lbl == 1 else (255, 0, 0)
            cv2.circle(display_img, (x, y), radius=5, color=color, thickness=-1)
            cv2.circle(display_img, (x, y), radius=5, color=(255, 255, 255), thickness=1)

        dpg.set_value(self._tag_texture, self.convert_to_texture(display_img, is_rgb=True))

    def _export_outputs(self) -> None:
        """Dispatch generated mask, composite overlay, ROI sample, and pair downstream."""
        with self._lock:
            mask, frame, raw_frame, pts = (
                self._current_mask,
                self._rgb_frame,
                self._frame,
                self.points.copy(),
            )
        if mask is None or frame is None or not pts:
            return
        if mask.shape[:2] != frame.shape[:2]:
            logger.warning(
                f"SAM: Mask shape {mask.shape[:2]} does not match frame shape {frame.shape[:2]}. Skipping export."
            )
            return

        mask_conns = self.connections.get("Mask", [])
        overlay_conns = self.connections.get("Overlay", [])
        roi_conns = self.connections.get("ROI", [])
        result_conns = self.connections.get("Result", [])

        mask_8u: Optional[np.ndarray] = None
        if mask_conns or roi_conns or result_conns:
            mask_8u = np.where(mask, np.uint8(255), np.uint8(0))

        # 1. Emit Mask
        if mask_conns and mask_8u is not None:
            for module in mask_conns:
                if hasattr(module, "input_cb"):
                    module.input_cb(data=mask_8u, data_type=IOTypes.FRAME)

        # 2. Emit Overlay
        if overlay_conns:
            overlay = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            masked_ov = overlay[mask]
            masked_ov >>= 1
            masked_ov[:, 2] += 127
            overlay[mask] = masked_ov
            for module in overlay_conns:
                if hasattr(module, "input_cb"):
                    module.input_cb(data=overlay, data_type=IOTypes.FRAME)

        # 3. Emit ROI Sample
        if roi_conns and mask_8u is not None:
            x, y, w, h = cv2.boundingRect(mask_8u)
            sample = {
                "uuid": self._roi_uuid,
                "name": "SAM_Object",
                "mask": mask,
                "rect": [x, self.texture_height - (y + h), w, h],
                "action": "create",
            }
            for module in roi_conns:
                if hasattr(module, "input_cb"):
                    module.input_cb(sample=sample, data_type=IOTypes.ROI_SAMPLE)

        # 4. Emit Result Pair (Frame + Mask)
        if result_conns and mask_8u is not None:
            out_frame = raw_frame.copy() if raw_frame is not None else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            result = (out_frame, mask_8u)
            for module in result_conns:
                if hasattr(module, "input_cb"):
                    module.input_cb(data=result, data_type=IOTypes.FRAME_MASK_PAIR)


EXPORTED_CLASS = SamInteractiveWin
EXPORTED_NAME = "SAM Interactive"
