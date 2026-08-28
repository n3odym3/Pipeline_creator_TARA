from __future__ import annotations
import dearpygui.dearpygui as dpg
from core.window_base import WindowBase
from core.input_output_types import IOTypes
from loguru import logger
import numpy as np
import cv2
import threading
from config.display_scaling import display_scaling
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# FluoSelectorWin
# ─────────────────────────────────────────────────────────────────────────────
class FluoSelectorWin(WindowBase):
    """
    Fluo Selector – FvFm Calculator.

    Receives two 12-bit (or 16-bit) frames:
      • F0  (via the 'F0'  input slot)
      • Fm  (via the 'Fm'  input slot, or the second call to input_cb
              when input_type == FRAME12 / FRAME16)

    Computes FvFm = (Fm - F0) / Fm  pixel-wise, then:
      1. Applies a light Gaussian blur to F0 and Fm before the ratio.
      2. Thresholds F0 pixels below `f0_min_pct`-th percentile → 0.
      3. Thresholds Fm pixels below `fm_min_pct`-th percentile → 0.
      4. Clips negative values to 0.
      5. Multiplies by 1 000 and casts to int16  (range 0-1 000).
      6. Clips values above `max_ratio * 1000` to 0 (removes aberrant noise).
      7. Emits the result as FRAME12 for downstream false-colour display.
    """

    accepted_input_types: list[Any]
    _persistent_fields: list[str]
    _export_folder: str | None
    _current_group: str
    _f0: np.ndarray | None
    _fm: np.ndarray | None
    _last_result: np.ndarray | None
    _last_display_result: np.ndarray | None
    _last_raw_result: np.ndarray | None
    _wait_f0: bool
    _wait_fm: bool
    _f0_fresh: bool
    _fm_fresh: bool
    _lock: threading.Lock
    _processing_lock: threading.Lock
    _is_comparing: bool
    _tag_f0_min: str
    _tag_fm_min: str
    _tag_max_ratio: str
    _tag_blur: str
    _tag_erosion: str
    _tag_dilation: str
    _tag_contours_on: str
    _tag_min_area: str
    _tag_wait_f0_btn: str
    _tag_wait_fm_btn: str
    _tag_compare_btn: str
    _tag_compute_btn: str
    _tag_clear_btn: str
    _tag_export_roi_btn: str
    _tag_export_roi_status: str
    _tag_sort_dir: str
    _tag_seg_mode: str
    _tag_start_point: str
    _tag_id_mode: str
    _tag_grid_cols: str
    _tag_grid_rows: str
    _tag_show_grid: str
    _theme_red: Any
    _theme_green: Any
    _theme_yellow: Any
    _last_contours: list[Any]
    _last_fvfm_shape: tuple[int, ...] | None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        label: str = "Fluo Selector",
        win_width: int = display_scaling.scale(350),
        win_height: int = display_scaling.scale(350),
        pos: tuple[int, int] = (50, 50),
        uuid: str | None = None,
        outputs: dict[str, Any] | None = None,
        visible: bool = True,
    ) -> None:

        outputs = outputs or {
            "FvFm": IOTypes.FRAME12,
            "ROI": IOTypes.ROI_SAMPLE,
        }

        super().__init__(label=label, pos=pos,
                         win_width=win_width, win_height=win_height,
                         uuid=uuid, outputs=outputs, visible=visible)

        self.outputs = outputs or {}
        self.connections = {k: [] for k in self.outputs}

        # Accepted inputs: F0 and Fm frames (12-bit or 16-bit HDR) + CMD_DICT for control
        self.accepted_input_types = [
            IOTypes.FRAME12,
            IOTypes.FRAME16,
            IOTypes.FRAME,
            IOTypes.CMD_DICT,
            IOTypes.FOLDER_PATH,
            IOTypes.TEXT,
        ]
        self._persistent_fields = ["label"]

        # Export settings
        self._export_folder = None
        self._current_group = "Unknown"

        # Internal frame storage
        self._f0 = None
        self._fm = None
        self._last_result = None
        self._last_display_result = None
        self._last_raw_result = None
        self._wait_f0 = False
        self._wait_fm = False
        self._f0_fresh = False
        self._fm_fresh = False
        self._lock = threading.Lock()
        self._processing_lock = threading.Lock()
        self._is_comparing = False

        # ── UI tags ───────────────────────────────────────────────────
        self._tag_f0_min      = f"fsel_f0_min_{self.UUID}"
        self._tag_fm_min      = f"fsel_fm_min_{self.UUID}"
        self._tag_max_ratio   = f"fsel_max_ratio_{self.UUID}"
        self._tag_blur        = f"fsel_blur_{self.UUID}"
        self._tag_erosion     = f"fsel_erosion_{self.UUID}"
        self._tag_dilation    = f"fsel_dilation_{self.UUID}"
        self._tag_contours_on = f"fsel_contours_on_{self.UUID}"
        self._tag_min_area    = f"fsel_min_area_{self.UUID}"
        self._tag_wait_f0_btn = f"fsel_wait_f0_{self.UUID}"
        self._tag_wait_fm_btn = f"fsel_wait_fm_{self.UUID}"
        self._tag_compare_btn = f"fsel_compare_btn_{self.UUID}"
        self._tag_compute_btn = f"fsel_compute_{self.UUID}"
        self._tag_clear_btn   = f"fsel_clear_{self.UUID}"
        self._tag_export_roi_btn = f"fsel_export_roi_{self.UUID}"
        self._tag_export_roi_status = f"fsel_export_roi_status_{self.UUID}"
        self._tag_sort_dir     = f"fsel_sort_dir_{self.UUID}"
        self._tag_seg_mode     = f"fsel_seg_mode_{self.UUID}"
        self._tag_start_point  = f"fsel_start_point_{self.UUID}"
        self._tag_id_mode      = f"fsel_id_mode_{self.UUID}"
        self._tag_grid_cols    = f"fsel_grid_cols_{self.UUID}"
        self._tag_grid_rows    = f"fsel_grid_rows_{self.UUID}"
        self._tag_show_grid    = f"fsel_show_grid_{self.UUID}"
        self._tag_show_id      = f"fsel_show_id_{self.UUID}"

        # Themes for status
        self._theme_red = None
        self._theme_green = None
        self._theme_yellow = None

        # Cache of last computed contours (for export)
        self._last_contours = []
        self._last_fvfm_shape = None

        self._build_ui()

        self.update_permission()

    def update_permission(self) -> None:
        """
        Adjust module permissions and UI visibility based on app_state.mode.
        Disable closing and lock all controls in user mode.
        """
        from core.app_state import app_state

        is_user = app_state.mode == "user"

        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # ── Themes ───────────────────────────────────────────────────
        with dpg.theme() as red_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (120, 40, 40))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (170, 60, 60))
        self._theme_red = red_theme

        with dpg.theme() as green_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (40, 120, 40))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (60, 170, 60))
        self._theme_green = green_theme

        with dpg.theme() as yellow_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (150, 150, 40))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (200, 200, 60))
        self._theme_yellow = yellow_theme

        with dpg.window(label=self.label,
                        width=self.win_width, height=self.win_height,
                        pos=self.pos, tag=self.winID, show=self.visible):

            # ── 1. Frame Status Buttons ───────────────────────────────
            with dpg.group(horizontal=True):
                dpg.add_button(label="WAIT F0", tag=self._tag_wait_f0_btn, 
                               callback=self._on_wait_f0, width=display_scaling.scale(150))
                dpg.add_button(label="WAIT FM", tag=self._tag_wait_fm_btn, 
                               callback=self._on_wait_fm, width=display_scaling.scale(150))
            
            # Initial state: Red
            dpg.bind_item_theme(self._tag_wait_f0_btn, self._theme_red)
            dpg.bind_item_theme(self._tag_wait_fm_btn, self._theme_red)

            # ── 2. Clear & Compute ─────────────────────────────────────
            with dpg.group(horizontal=True):
                dpg.add_button(label="Clear", tag=self._tag_clear_btn,
                               callback=self._on_clear, width=display_scaling.scale(150))
                dpg.add_button(label="Compute FvFm", tag=self._tag_compute_btn,
                               callback=self._on_compute, width=display_scaling.scale(150))

            dpg.add_spacer(height=display_scaling.scale(5))
            dpg.add_separator()
            dpg.add_spacer(height=display_scaling.scale(5))

            # ── 3. Main Controls ──────────────────────────────────────
            # "Slider supérieur" is likely Fm min % since it defines the signal peak
            with dpg.group():
                with dpg.group(horizontal=True):
                    dpg.add_text("Thresh %", color=(180, 180, 180))
                    dpg.add_button(label="COMPARE",
                                tag=self._tag_compare_btn,
                                width=display_scaling.scale(200))
                dpg.add_drag_float(
                    label="",
                    tag=self._tag_fm_min,
                    default_value=90.0,
                    min_value=80.0, max_value=100.0,
                    format="%.1f",
                    speed=0.05,
                    width=-1,
                    callback=self._on_compute,
                )
                with dpg.item_handler_registry() as thresh_fm_handler:
                    dpg.add_item_deactivated_after_edit_handler(callback=self._on_compute)
                dpg.bind_item_handler_registry(self._tag_fm_min, thresh_fm_handler)

            dpg.add_spacer(height=display_scaling.scale(5))

            # Contours & Filter
            with dpg.group():
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(
                        label="Detect",
                        tag=self._tag_contours_on,
                        default_value=True,
                        callback=self._on_compute
                    )

                    dpg.add_text("Thresh", color=(180, 180, 180))
                    dpg.add_drag_int(
                        label="",
                        tag=self._tag_min_area,
                        default_value=2500,
                        min_value=0,
                        max_value=15000,
                        speed=100,
                        width=-1,
                        callback=self._on_compute
                    )

                with dpg.item_handler_registry() as area_handler:
                    dpg.add_item_deactivated_after_edit_handler(callback=self._on_compute)
                dpg.bind_item_handler_registry(self._tag_min_area, area_handler)
            
            with dpg.group(horizontal=True):

                dpg.add_button(label="Export",
                            tag=self._tag_export_roi_btn,
                            callback=self._export_contour_rois,
                            width=display_scaling.scale(150))

                dpg.add_text("", tag=self._tag_export_roi_status, color=(140, 200, 140))

            # ── 4. Advanced (Hidden) ──────────────────────────────────
            dpg.add_spacer(height=display_scaling.scale(10))
            with dpg.collapsing_header(label="Advanced Options",default_open=True):
                # ── Segmentation Mode (fallback when FvFm signal is too weak) ──
                dpg.add_text("Segmentation Source", color=(130, 130, 130))
                dpg.add_combo(
                    items=["FvFm", "F0", "Fm"],
                    tag=self._tag_seg_mode,
                    default_value="FvFm",
                    width=-1,
                    callback=self._on_compute
                )

                dpg.add_spacer(height=display_scaling.scale(4))
                dpg.add_text("Export Sorting Strategy", color=(130, 130, 130))
                dpg.add_radio_button(
                    items=["Column","Row"],
                    tag=self._tag_sort_dir,
                    default_value="Column",
                    horizontal=True
                )

                dpg.add_spacer(height=display_scaling.scale(4))
                dpg.add_text("Starting Point", color=(130, 130, 130))
                dpg.add_combo(
                    items=["Top Left", "Top Right", "Bottom Left", "Bottom Right"],
                    tag=self._tag_start_point,
                    default_value="Top Left",
                    width=-1
                )

                dpg.add_spacer(height=display_scaling.scale(4))
                dpg.add_text("ID Mode", color=(130, 130, 130))
                dpg.add_combo(
                    items=["Grid ID", "Absolute ID"],
                    tag=self._tag_id_mode,
                    default_value="Grid ID",
                    width=-1
                )

                dpg.add_spacer(height=display_scaling.scale(4))
                dpg.add_checkbox(
                    label="Show Grid",
                    tag=self._tag_show_grid,
                    default_value=False,
                    callback=self._on_compute
                )

                dpg.add_checkbox(
                    label="Show ID",
                    tag=self._tag_show_id,
                    default_value=False,
                    callback=self._on_compute
                )

                with dpg.group(horizontal=True):
                    dpg.add_text("Grid Cols", color=(130, 130, 130))
                    dpg.add_input_int(
                        label="",
                        tag=self._tag_grid_cols,
                        default_value=4,
                        min_value=1,
                        width=display_scaling.scale(100),
                        callback=self._on_compute
                    )
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Rows", color=(130, 130, 130))
                    dpg.add_input_int(
                        label="",
                        tag=self._tag_grid_rows,
                        default_value=5,
                        min_value=1,
                        width=display_scaling.scale(100),
                        callback=self._on_compute
                    )
                with dpg.item_handler_registry() as grid_handler:
                    dpg.add_item_deactivated_after_edit_handler(callback=self._on_compute)
                dpg.bind_item_handler_registry(self._tag_grid_cols, grid_handler)
                dpg.bind_item_handler_registry(self._tag_grid_rows, grid_handler)

                dpg.add_spacer(height=display_scaling.scale(4))
                dpg.add_text("Blur kernel", color=(130, 130, 130))
                dpg.add_slider_int(
                    label="",
                    tag=self._tag_blur,
                    default_value=0,
                    min_value=0, max_value=21,
                    width=-1,
                    callback=self._on_compute
                )

                dpg.add_text("Dilation (Merge)", color=(130, 130, 130))
                dpg.add_slider_int(
                    label="",
                    tag=self._tag_dilation,
                    default_value=0,
                    min_value=0, max_value=20,
                    width=-1,
                    callback=self._on_compute
                )

                dpg.add_text("Erosion (Noise)", color=(130, 130, 130))
                dpg.add_slider_int(
                    label="",
                    tag=self._tag_erosion,
                    default_value=0,
                    min_value=0, max_value=20,
                    width=-1,
                    callback=self._on_compute
                )

                dpg.add_text("Max FvFm Gate", color=(130, 130, 130))
                dpg.add_input_float(
                    label="",
                    tag=self._tag_max_ratio,
                    default_value=0.9,
                    min_value=0.0, max_value=1.0,
                    step=-1,
                    width=-1,
                    callback=self._on_compute
                )

                with dpg.group():
                    dpg.add_text("F0 raw mask %", color=(130, 130, 130))
                    dpg.add_drag_int(
                        label="",
                        tag=self._tag_f0_min,
                        default_value=25,
                        min_value=0, max_value=100,
                        width=-1,
                        callback=self._on_compute,
                        speed=0.25
                    )
                    with dpg.item_handler_registry() as thresh_f0_handler:
                        dpg.add_item_deactivated_after_edit_handler(callback=self._on_compute)
                    dpg.bind_item_handler_registry(self._tag_f0_min, thresh_f0_handler)

        # ── Setup Compare Handlers ──────────────────────────────────
        with dpg.item_handler_registry(tag=f"fsel_comp_hand_{self.UUID}"):
            dpg.add_item_clicked_handler(callback=self._compare_on)
        dpg.bind_item_handler_registry(self._tag_compare_btn, f"fsel_comp_hand_{self.UUID}")
        
        # We use a global handler for release to ensure it triggers even if mouse 
        # leaves the button area
        with dpg.handler_registry():
            dpg.add_mouse_release_handler(callback=self._compare_off)

    # ------------------------------------------------------------------
    # Input callback
    # ------------------------------------------------------------------
    def input_cb(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Receives frames, commands, folder paths or group names.
        """
        # --- Handle Folder and Group ---
        data_type = kwargs.get("data_type")
        if data_type == IOTypes.FOLDER_PATH:
            self._export_folder = data
            return
        elif data_type == IOTypes.TEXT:
            self._current_group = data
            return
        elif isinstance(data, str):
            import os
            if os.path.isdir(data):
                self._export_folder = data
            else:
                self._current_group = data
            return

        # --- Handle CMD_DICT (e.g. for clearing from other modules) ---
        if data_type == IOTypes.CMD_DICT or isinstance(data, dict):
            cmd = data if isinstance(data, dict) else kwargs.get("data")
            if cmd and cmd.get("action") == "clear":
                self._on_clear()
            return

        # --- Handle Frames ---
        frame = self._extract_frame(data, kwargs)
        if frame is None:
            return

        slot = kwargs.get("slot", "").upper()
        trigger_compute = False

        with self._lock:
            # Determine which bucket this frame belongs to
            is_f0 = False
            is_fm = False
            
            # 1. Explicit slot priority
            if slot == "F0":
                is_f0 = True
            elif slot in ["FM", "Fm"]:
                is_fm = True
            # 2. Waiting priority
            elif self._wait_f0:
                is_f0 = True
            elif self._wait_fm:
                is_fm = True
            # 3. Initial placement
            elif slot == "" and self._f0 is None:
                is_f0 = True
            elif slot == "" and self._fm is None:
                is_fm = True

            if is_f0:
                was_waiting = self._wait_f0 or (self._f0 is None)
                self._f0 = frame.copy()
                self._wait_f0 = False
                if was_waiting:
                    self._f0_fresh = True
                self._update_btn_status(self._tag_wait_f0_btn, received=True)
            elif is_fm:
                was_waiting = self._wait_fm or (self._fm is None)
                self._fm = frame.copy()
                self._wait_fm = False
                if was_waiting:
                    self._fm_fresh = True
                self._update_btn_status(self._tag_wait_fm_btn, received=True)
            else:
                # No one is waiting for this frame and slots don't match or are full.
                return

            # Auto-compute ONLY if both frames are present AND both have been 'captured' 
            # (either via a 'Wait' or as the initial pair after a Clear).
            if self._f0 is not None and self._fm is not None:
                if self._f0_fresh and self._fm_fresh:
                    trigger_compute = True
                    self._f0_fresh = False
                    self._fm_fresh = False

        if trigger_compute:
            self._on_compute()

    def _compare_on(self, *args: Any, **kwargs: Any) -> None:
        """Shows Fm when button is clicked."""
        with self._lock:
            self._is_comparing = True
            if self._fm is None:
                return
            frame_to_show = self._fm.copy()
            
        # Ensure it's in uint16 for FRAME12 output
        if frame_to_show.dtype != np.uint16:
            # If float or wider, scale/clip to 12-bit range (0-4095) for consistency
            if frame_to_show.max() > 4095:
                # Basic scaling if it's 16-bit
                frame_to_show = (frame_to_show / 16).astype(np.uint16)
            else:
                frame_to_show = frame_to_show.astype(np.uint16)
                
        self._emit_frame(frame_to_show)

    def _compare_off(self, *args: Any, **kwargs: Any) -> None:
        """Reverts to processed FvFm when mouse button is released."""
        with self._lock:
            if not self._is_comparing:
                return
            self._is_comparing = False
            
            if self._last_display_result is not None:
                frame_to_show = self._last_display_result.copy()
            elif self._last_result is not None:
                frame_to_show = self._last_result.copy()
            else:
                return
        
        self._emit_frame(frame_to_show)

    def _emit_frame(self, frame: np.ndarray) -> None:
        for module in self.connections.get("FvFm", []):
            if hasattr(module, "input_cb"):
                try:
                    module.input_cb(data=frame, data_type=IOTypes.FRAME12)
                except Exception as e:
                    logger.error(f"FluoSelector compare emit error: {e}")

    def _export_contour_rois(self, *args: Any, **kwargs: Any) -> None:
        """Export using the received export folder path, or prompt if not set."""
        contours = self._last_contours
        shape = self._last_fvfm_shape

        if not contours or shape is None:
            dpg.set_value(self._tag_export_roi_status, "No contours to export.")
            logger.warning("FluoSelector: no contours to export.")
            return

        if self._export_folder:
            self._do_export_contour_rois()
        else:
            from core.file_explorer import file_explorer
            default_dir = ""

            def _on_folder_selected(folder_path: str, *a, **kw):
                if not folder_path:
                    dpg.set_value(self._tag_export_roi_status, "Export cancelled.")
                    return
                    
                self._export_folder = folder_path
                self._do_export_contour_rois()

            file_explorer.select_folder(default_path=default_dir, callback=_on_folder_selected)

    def _do_export_contour_rois(self) -> None:
        """Export each valid contour as a mask-based ROI_SAMPLE."""
        import uuid as uuid_mod

        contours = self._last_contours
        shape = self._last_fvfm_shape

        # --- 1. Send Clear action first to reset downstream ROI managers ---
        clear_sample = {"action": "clear", "uuid": "clear_all"}
        for module in self.connections.get("ROI", []):
            if hasattr(module, "input_cb"):
                try:
                    module.input_cb(sample=clear_sample, data_type=IOTypes.ROI_SAMPLE)
                except Exception as e:
                    logger.error(f"FluoSelector ROI clear error: {e}")

        # --- 2. Sort contours based on spatial position (Adaptive Sequential Grouping) ---
        sort_strategy = dpg.get_value(self._tag_sort_dir)
        
        # Calculate centroids and dimensions
        contour_data = []
        sum_w, sum_h = 0.0, 0.0
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            sum_w += bw
            sum_h += bh
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = bx + bw // 2, by + bh // 2
            contour_data.append({"cnt": cnt, "x": cx, "y": cy, "w": bw, "h": bh})
        
        num = len(contour_data)
        if num == 0:
            return
            
        avg_w = sum_w / num
        avg_h = sum_h / num

        # Get starting point configuration
        start_point = dpg.get_value(self._tag_start_point)
        
        final_sorted = []
        if sort_strategy == "Row":
            # Determine Y sort direction based on starting point
            reverse_y = start_point in ["Bottom Left", "Bottom Right"]
            # Determine X sort direction based on starting point
            reverse_x = start_point in ["Top Right", "Bottom Right"]
            
            # 1. Sort by Y according to starting point
            contour_data.sort(key=lambda c: c["y"], reverse=reverse_y)
            # 2. Group into rows using a dynamic threshold (70% of avg height)
            rows = []
            if contour_data:
                curr_row = [contour_data[0]]
                for i in range(1, len(contour_data)):
                    # Compare to average Y of current row for better tolerance
                    avg_row_y = sum(c["y"] for c in curr_row) / len(curr_row)
                    if abs(contour_data[i]["y"] - avg_row_y) < (avg_h * 0.7):
                        curr_row.append(contour_data[i])
                    else:
                        rows.append(curr_row)
                        curr_row = [contour_data[i]]
                rows.append(curr_row)
            # 3. Sort each row by X according to starting point and flatten
            for r in rows:
                r.sort(key=lambda c: c["x"], reverse=reverse_x)
                final_sorted.extend(r)
        else:
            # Determine X sort direction based on starting point
            reverse_x = start_point in ["Top Right", "Bottom Right"]
            # Determine Y sort direction based on starting point
            reverse_y = start_point in ["Bottom Left", "Bottom Right"]
            
            # 1. Sort by X according to starting point
            contour_data.sort(key=lambda c: c["x"], reverse=reverse_x)
            # 2. Group into columns using a dynamic threshold (70% of avg width)
            cols = []
            if contour_data:
                curr_col = [contour_data[0]]
                for i in range(1, len(contour_data)):
                    # Compare to average X of current column for better tolerance
                    avg_col_x = sum(c["x"] for c in curr_col) / len(curr_col)
                    if abs(contour_data[i]["x"] - avg_col_x) < (avg_w * 0.7):
                        curr_col.append(contour_data[i])
                    else:
                        cols.append(curr_col)
                        curr_col = [contour_data[i]]
                cols.append(curr_col)
            # 3. Sort each column by Y according to starting point and flatten
            for c in cols:
                c.sort(key=lambda item: item["y"], reverse=reverse_y)
                final_sorted.extend(c)
            
        sorted_contours = final_sorted

        # --- 3. Export each contour ---
        img_h, img_w = shape[:2]
        exported = 0
        id_mode = dpg.get_value(self._tag_id_mode)
        num_cols = max(1, dpg.get_value(self._tag_grid_cols))
        num_rows = max(1, dpg.get_value(self._tag_grid_rows))
        seen_slots = {}

        for i, item in enumerate(sorted_contours):
            cnt = item["cnt"]
            cx = item["x"]
            cy = item["y"]

            # Build binary mask filled inside the contour
            roi_mask_uint8 = np.zeros(shape, dtype=np.uint8)
            cv2.drawContours(roi_mask_uint8, [cnt], -1, 1, -1)  # filled
            roi_mask = roi_mask_uint8.astype(bool)

            # Compute bounding rectangle (in image coords: top-left origin)
            bx, by, bw, bh = cv2.boundingRect(cnt)

            # Convert to DPG plot coordinate system (origin = bottom-left):
            #   plot_x = image_x
            #   plot_y (bottom of rect) = img_h - (image_y + image_h)
            rect_x = bx
            rect_y = img_h - (by + bh)   # bottom-left corner in plot space

            # Compute ROI name based on ID mode
            if id_mode == "Grid ID":
                cell_w = img_w / num_cols
                cell_h = img_h / num_rows
                
                col_idx = min(max(0, int(cx // cell_w)), num_cols - 1)
                row_idx = min(max(0, int(cy // cell_h)), num_rows - 1)
                
                # Logical coordinates relative to starting point
                if start_point == "Top Left":
                    c = col_idx
                    r = row_idx
                elif start_point == "Top Right":
                    c = (num_cols - 1) - col_idx
                    r = row_idx
                elif start_point == "Bottom Left":
                    c = col_idx
                    r = (num_rows - 1) - row_idx
                elif start_point == "Bottom Right":
                    c = (num_cols - 1) - col_idx
                    r = (num_rows - 1) - row_idx
                else:
                    c = col_idx
                    r = row_idx
                    
                if sort_strategy == "Row":
                    slot_id = r * num_cols + c + 1
                else:  # Column
                    slot_id = c * num_rows + r + 1
                    
                if slot_id in seen_slots:
                    seen_slots[slot_id] += 1
                    roi_name = f"{slot_id}_{seen_slots[slot_id]}"
                else:
                    seen_slots[slot_id] = 1
                    roi_name = f"{slot_id}"
            else:
                roi_name = f"{i+1}"

            # Stable UUID based on module UUID + current group + contour ROI name
            roi_uuid = str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS,
                                          f"{self.UUID}_{self._current_group}_contour_{roi_name}"))

            sample = {
                "uuid": roi_uuid,
                "name": roi_name,
                "mask": roi_mask,          # bool ndarray for mean calculation
                "rect": [rect_x, rect_y, bw, bh],  # enclosing rect for viewer display
                "action": "create",
            }

            for module in self.connections.get("ROI", []):
                if hasattr(module, "input_cb"):
                    try:
                        module.input_cb(sample=sample, data_type=IOTypes.ROI_SAMPLE)
                    except Exception as e:
                        logger.error(f"FluoSelector ROI export error: {e}")
            exported += 1

        msg = f"{exported} contour(s) exported as ROIs."
        dpg.set_value(self._tag_export_roi_status, msg)
        logger.success(f"FluoSelector: {msg}")

        # --- 4. Export FVFM Data ---
        if self._export_folder and self._last_result is not None:
            import os
            try:
                os.makedirs(self._export_folder, exist_ok=True)
                
                # To make the 16-bit TIFF visible in standard viewers while keeping the decimal value recoverable:
                # Internal values are scaled by 1000 (FvFm * 1000). 
                # We scale them by an additional 65.535 to use the full 16-bit dynamic range (0-65535).
                # To recover the real FvFm ratio later, simply divide the pixel value by 65535.0.
                scale_factor = 65535.0 / 1000.0
                
                # Save Processed
                processed_export = np.clip(self._last_result.astype(np.float32) * scale_factor, 0, 65535).astype(np.uint16)
                processed_filename = f"{self._current_group}_FvFm_Processed.tiff"
                processed_filepath = os.path.join(self._export_folder, processed_filename)
                cv2.imwrite(processed_filepath, processed_export)
                
                # Save Raw
                if getattr(self, "_last_raw_result", None) is not None:
                    raw_export = np.clip(self._last_raw_result.astype(np.float32) * scale_factor, 0, 65535).astype(np.uint16)
                    raw_filename = f"{self._current_group}_FvFm_Raw.tiff"
                    raw_filepath = os.path.join(self._export_folder, raw_filename)
                    cv2.imwrite(raw_filepath, raw_export)
                    
                logger.success(f"FluoSelector: Saved FvFm Raw and Processed data to {self._export_folder}")
            except Exception as e:
                logger.error(f"FluoSelector: Failed to save FvFm data: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_frame(data: Any, kwargs: dict[str, Any]) -> np.ndarray | None:
        """Pull a numpy array out of the various possible input shapes."""
        frame = None
        if isinstance(data, np.ndarray):
            frame = data
        elif isinstance(data, (tuple, list)) and len(data) >= 1:
            if isinstance(data[0], np.ndarray):
                frame = data[0]
        else:
            # Fallback: look in kwargs
            f = kwargs.get("frame") if kwargs.get("frame") is not None else kwargs.get("data")
            if isinstance(f, np.ndarray):
                frame = f

        if frame is not None:
            # Ignore RGB / multi-channel frames to prevent OpenCV findContours crashes
            if len(frame.shape) > 2:
                logger.warning(f"FluoSelector: Ignored RGB/multi-channel frame with shape {frame.shape}")
                return None
            return frame

        logger.warning("FluoSelector: could not extract frame from input")
        return None

    def _update_btn_status(self, tag: str, received: bool = True) -> None:
        theme = self._theme_green if received else self._theme_red
        dpg.bind_item_theme(tag, theme)

    def _on_wait_f0(self, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            self._wait_f0 = True
            self._f0_fresh = False
        dpg.bind_item_theme(self._tag_wait_f0_btn, self._theme_yellow)

    def _on_wait_fm(self, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            self._wait_fm = True
            self._fm_fresh = False
        dpg.bind_item_theme(self._tag_wait_fm_btn, self._theme_yellow)

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------
    def _on_compute(self, *args: Any, **kwargs: Any) -> None:
        """Triggers the computation thread if not already processing."""
        if self._processing_lock.locked():
            # logger.debug("FluoSelector: compute already in progress, skipping.")
            return
        threading.Thread(target=self._compute_worker, daemon=True).start()

    def _compute_worker(self) -> None:
        """Worker thread that executes the actual compute while holding the processing lock."""
        # Note: blocking=False is used here as a secondary safeguard, 
        # but _on_compute check should have already caught it.
        if not self._processing_lock.acquire(blocking=False):
            return
        try:
            self._compute()
        finally:
            self._processing_lock.release()

    def _compute(self) -> None:
        # ── Determine requirements based on mode ──
        seg_mode = dpg.get_value(self._tag_seg_mode)
        
        with self._lock:
            f0_raw = self._f0.copy() if self._f0 is not None else None
            fm_raw = self._fm.copy() if self._fm is not None else None

        # Check if we have the necessary frames
        if seg_mode == "FvFm":
            if f0_raw is None or fm_raw is None:
                return
            # Update button label
            dpg.set_item_label(self._tag_compute_btn, "Compute FvFm")
        elif seg_mode == "F0":
            if f0_raw is None:
                return
            dpg.set_item_label(self._tag_compute_btn, "Process F0")
        elif seg_mode == "Fm":
            if fm_raw is None:
                return
            dpg.set_item_label(self._tag_compute_btn, "Process Fm")

        # ── Read UI params ───────────────────────────────────────────
        main_thresh_pct = dpg.get_value(self._tag_fm_min)   # MAIN slider
        f0_mask_pct     = dpg.get_value(self._tag_f0_min)   # ADVANCED slider
        max_ratio       = dpg.get_value(self._tag_max_ratio)
        blur_k          = dpg.get_value(self._tag_blur)
        erosion_it      = dpg.get_value(self._tag_erosion)
        dilation_it     = dpg.get_value(self._tag_dilation)
        contours_on     = dpg.get_value(self._tag_contours_on)
        min_area        = dpg.get_value(self._tag_min_area)

        # ── Processing ───────────────────────────────────────────────
        # Prep working copies as float32
        f0 = f0_raw.astype(np.float32) if f0_raw is not None else None
        fm = fm_raw.astype(np.float32) if fm_raw is not None else None

        # ── Calculate RAW FvFm ──
        raw_result = None
        if f0 is not None and fm is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                raw_fvfm = np.where(fm > 0, (fm - f0) / fm, 0.0)
            np.clip(raw_fvfm, 0.0, None, out=raw_fvfm)
            raw_result = (raw_fvfm * 1000.0).astype(np.uint16)

        # Blur
        if blur_k > 0:
            k = int(blur_k) | 1
            if f0 is not None: f0 = cv2.GaussianBlur(f0, (k, k), 0)
            if fm is not None: fm = cv2.GaussianBlur(fm, (k, k), 0)

        # ── Mode Selection Logic ─────────────────────────────────────
        if seg_mode == "F0":
            # 1. Mask using Main Slider on F0
            thresh_val = np.percentile(f0, main_thresh_pct) if main_thresh_pct > 0 else 0
            seg_mask = f0 < thresh_val
            # 2. Result is F0 scaled to 0-1000 for display consistency
            f_max = f0.max()
            if f_max > 0:
                fvfm_int = (f0 / f_max * 1000.0).astype(np.int32)
            else:
                fvfm_int = np.zeros_like(f0, dtype=np.int32)
            
        elif seg_mode == "Fm":
            # 1. Mask using Main Slider on Fm
            thresh_val = np.percentile(fm, main_thresh_pct) if main_thresh_pct > 0 else 0
            seg_mask = fm < thresh_val
            # 2. Result is Fm scaled to 0-1000
            f_max = fm.max()
            if f_max > 0:
                fvfm_int = (fm / f_max * 1000.0).astype(np.int32)
            else:
                fvfm_int = np.zeros_like(fm, dtype=np.int32)

        else: # "FvFm"
            # 1. Dual masks for calculation safety
            f0_m = (f0 < np.percentile(f0, f0_mask_pct)) if f0_mask_pct > 0 else np.zeros(f0.shape, dtype=bool)
            fm_m = (fm < np.percentile(fm, main_thresh_pct)) if main_thresh_pct > 0 else np.zeros(fm.shape, dtype=bool)
            seg_mask = f0_m | fm_m
            
            # 2. FvFm Ratio
            with np.errstate(divide='ignore', invalid='ignore'):
                fvfm = np.where(fm > 0, (fm - f0) / fm, 0.0)
            np.clip(fvfm, 0.0, None, out=fvfm)
            fvfm[seg_mask] = 0.0
            
            # 3. Scale and Noise Gate
            fvfm_int = (fvfm * 1000.0).astype(np.int32)
            noise_limit = int(max_ratio * 1000)
            fvfm_int[fvfm_int > noise_limit] = 0
            
        # ── Morphology & Cleanup ─────────────────────────────────────
        if erosion_it > 0 or dilation_it > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask_uint = seg_mask.astype(np.uint8)
            if erosion_it > 0: mask_uint = cv2.dilate(mask_uint, kernel, iterations=erosion_it)
            if dilation_it > 0: mask_uint = cv2.erode(mask_uint, kernel, iterations=dilation_it)
            seg_mask = mask_uint > 0

        # Final zero-out on display image based on (morphed) mask
        fvfm_int[seg_mask] = 0
        np.clip(fvfm_int, 0, None, out=fvfm_int)

        # ── Contour Detection & Strict Masking ────────────────────────
        self._last_contours = []
        self._last_fvfm_shape = fvfm_int.shape
        if contours_on:
            is_dragging = dpg.is_item_active(self._tag_fm_min)
            if not is_dragging:
                # 1. Find potential contours on the exclusion mask
                mask_8u = (~seg_mask).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 2. Filter by area
                valid_contours = []
                for cnt in contours:
                    if cv2.contourArea(cnt) >= min_area:
                        valid_contours.append(cnt)
                self._last_contours = valid_contours

                # 3. Create a binary mask of ONLY valid contours (filled)
                final_roi_mask = np.zeros(fvfm_int.shape, dtype=np.uint8)
                if valid_contours:
                    cv2.drawContours(final_roi_mask, valid_contours, -1, 255, -1) # -1 = filled
                
                # 4. Strict Enforcement: Hide everything not inside a valid contour
                fvfm_int[final_roi_mask == 0] = 0
                
                dpg.set_value(self._tag_export_roi_status, f"{len(self._last_contours)} ROI(s) [{seg_mode}]")
            else:
                pass
        else:
            # Contours off
            dpg.set_value(self._tag_export_roi_status, "")

        # Cast to uint16 for storage (CLEAN version)
        result_to_store = fvfm_int.astype(np.uint16)

        # Prepare display result (WITH outlines for visual feedback)
        result_to_emit = result_to_store.copy()
        if contours_on and not dpg.is_item_active(self._tag_fm_min):
            for cnt in self._last_contours:
                cv2.drawContours(result_to_emit, [cnt], -1, 1000, 1) # Outline

        # Draw grid lines / IDs if requested
        show_grid = dpg.get_value(self._tag_show_grid)
        show_id = dpg.get_value(self._tag_show_id)
        if (show_grid or show_id) and not dpg.is_item_active(self._tag_fm_min):
            h, w = result_to_emit.shape[:2]
            num_cols = max(1, dpg.get_value(self._tag_grid_cols))
            num_rows = max(1, dpg.get_value(self._tag_grid_rows))
            cell_w = w / num_cols
            cell_h = h / num_rows
            
            if show_grid:
                # Draw vertical lines
                for col in range(1, num_cols):
                    x = int(col * w / num_cols)
                    cv2.line(result_to_emit, (x, 0), (x, h), 700, 1)
                    
                # Draw horizontal lines
                for row in range(1, num_rows):
                    y = int(row * h / num_rows)
                    cv2.line(result_to_emit, (0, y), (w, y), 700, 1)
            
            if show_id:
                start_point = dpg.get_value(self._tag_start_point)
                sort_strategy = dpg.get_value(self._tag_sort_dir)
                for col_idx in range(num_cols):
                    for row_idx in range(num_rows):
                        # Determine coordinates relative to starting point
                        if start_point == "Top Left":
                            c = col_idx
                            r = row_idx
                        elif start_point == "Top Right":
                            c = (num_cols - 1) - col_idx
                            r = row_idx
                        elif start_point == "Bottom Left":
                            c = col_idx
                            r = (num_rows - 1) - row_idx
                        elif start_point == "Bottom Right":
                            c = (num_cols - 1) - col_idx
                            r = (num_rows - 1) - row_idx
                        else:
                            c = col_idx
                            r = row_idx
                            
                        if sort_strategy == "Row":
                            slot_id = r * num_cols + c + 1
                        else: # Column
                            slot_id = c * num_rows + r + 1
                            
                        text = str(slot_id)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1
                        thickness = 1
                        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                        
                        tx = int((col_idx + 1) * cell_w - text_size[0] - 5)
                        ty = int((row_idx + 1) * cell_h - 5)
                        
                        # Draw black shadow for readability
                        cv2.putText(result_to_emit, text, (tx + 1, ty + 1), font, font_scale, 0, thickness + 1, cv2.LINE_AA)
                        # Draw orange/yellow text (value 850 maps to orange in Inferno colormap)
                        cv2.putText(result_to_emit, text, (tx, ty), font, font_scale, 850, thickness, cv2.LINE_AA)

        with self._lock:
            self._last_result = result_to_store.copy()
            self._last_display_result = result_to_emit.copy()
            if raw_result is not None:
                self._last_raw_result = raw_result.copy()

        # ── Emit ─────────────────────────────────────────────────────
        for module in self.connections.get("FvFm", []):
            if hasattr(module, "input_cb"):
                try:
                    module.input_cb(data=result_to_emit, data_type=IOTypes.FRAME12)
                except Exception as e:
                    logger.error(f"FluoSelector emit error: {e}")

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------
    def _on_clear(self, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            self._f0 = None
            self._fm = None
            self._last_result = None
            self._last_display_result = None
            self._last_raw_result = None
            self._wait_f0 = False
            self._wait_fm = False
            self._f0_fresh = False
            self._fm_fresh = False
        self._update_btn_status(self._tag_wait_f0_btn, received=False)
        self._update_btn_status(self._tag_wait_fm_btn, received=False)

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------
    def is_ready(self) -> bool:
        return True

    def is_outputs_ready(self) -> bool:
        return True


# ─────────────────────────────────────────────────────────────────────────────
EXPORTED_CLASS = FluoSelectorWin
EXPORTED_NAME  = "Fluo Selector (FvFm)"
