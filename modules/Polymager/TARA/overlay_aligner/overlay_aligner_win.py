"""
Overlay Aligner window module.

Superimposes two images (A and B) of possibly different resolution / bit depth /
zoom / lens distortion, lets the user align them manually or from picked point
pairs, blends B over A (opacity, colorization, transparency thresholds) and emits
the result as an RGBA (BGRA, uint8) frame. All alignment settings can be saved
to / loaded from JSON.

The heavy lifting lives in :mod:`overlay_engine` (GUI-free, unit-testable).
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
from loguru import logger

from config.display_scaling import display_scaling
from core.input_output_types import IOTypes
from core.window_base import WindowBase
from modules.Polymager.TARA.overlay_aligner import overlay_engine as eng

IMAGE_EXTENSIONS = [
    ("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
    ("All files", "*.*"),
]


class OverlayAlignerWin(WindowBase):
    """
    Overlay Aligner – align and blend two images into one RGBA frame.

    Inputs (first free slot, or the one armed with WAIT A / WAIT B):
      • Layer A: reference image (typically an RGB camera frame).
      • Layer B: image placed over A (RGB, or grayscale 8/12/16-bit such as the
        FvFm frame from Fluo Selector).

    Alignment tools: zoom, aspect, rotation, shift, flip, radial (spherical)
    distortion correction, plus a semi-manual point-pair fit (translation,
    similarity, affine, homography). Colorization (palette / min / max) and
    transparency thresholds are available on both layers. Output: RGBA frame
    (BGRA uint8 with alpha, black + transparent outside the images).
    """

    A_COLOR = (80, 220, 255, 255)
    B_COLOR = (255, 90, 200, 255)
    PAIR_COLOR = (255, 220, 60, 160)
    CANVAS_COLOR = (200, 200, 200, 140)
    GRID_COLOR = (255, 255, 255, 70)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        label: str = "Overlay Aligner",
        win_width: int = display_scaling.scale(1400),
        win_height: int = display_scaling.scale(900),
        pos: tuple[int, int] = (60, 60),
        uuid: str | None = None,
        outputs: dict[str, Any] | None = None,
        visible: bool = True,
        settings_json: str = "",
        **kwargs: Any,
    ) -> None:
        settings_json = kwargs.get("settings_json", settings_json)
        outputs = outputs or {"RGBA": IOTypes.FRAME}
        super().__init__(
            label=label, pos=pos, win_width=win_width, win_height=win_height,
            uuid=uuid, outputs=outputs, visible=visible, **kwargs,
        )
        self.outputs = outputs
        self.connections = {k: [] for k in self.outputs}
        self.accepted_input_types = [
            IOTypes.FRAME,
            IOTypes.FRAME12,
            IOTypes.FRAME16,
            IOTypes.FRAME_MASK_PAIR,
            IOTypes.FILE_PATH,
            IOTypes.FOLDER_PATH,
            IOTypes.TEXT,
            IOTypes.CMD_DICT,
        ]
        self.persistent_fields = ["label", "settings_json"]

        # ── State ────────────────────────────────────────────────────
        self._lock = threading.RLock()
        self.layers: dict[str, eng.Layer] = {"A": eng.Layer("A"), "B": eng.Layer("B")}
        self.g: dict[str, Any] = eng.default_global_params()
        self._armed: str | None = None
        self._fresh = {"A": False, "B": False}
        self._export_folder: str | None = None
        self._group_name = "Overlay"
        self._last_fit_msg = ""

        self.running = True
        self._ui_ready = False
        self._dirty = threading.Event()
        self._render_cache: dict[str, Any] = {}
        self._emit_lock = threading.Lock()
        self._emit_timer: threading.Timer | None = None
        self._fit_pending = True
        self._overlay_items: list[Any] = []
        self._selected_point: tuple[str, int] | None = None
        self._last_canvas: tuple[int, int, int, int] = (0, 0, 640, 480)
        self._tex_w = 0
        self._tex_h = 0
        self._saved_settings: str = settings_json
        self._current_rendered_bgr: np.ndarray | None = None
        self._current_rendered_a: np.ndarray | None = None
        self._current_rendered_pw: int = 0
        self._current_rendered_ph: int = 0
        self._present_pending: bool = False
        self._pending_frame: tuple[np.ndarray, int, int, int, int] | None = None

        # ── Tags ─────────────────────────────────────────────────────
        self._plot = self._t("plot")
        self._xaxis = self._t("xaxis")
        self._yaxis = self._t("yaxis")
        self._series = self._t("series")
        self._tex = self._t("texture")
        self._tex_reg = self._t("texreg")
        self._ctl_w = display_scaling.scale(190)
        self._panel_w = display_scaling.scale(420)

        self._theme_red: Any = None
        self._theme_green: Any = None
        self._theme_yellow: Any = None

        self._build_ui()
        self._ui_ready = True

        # Do not reload settings by default; user can click 'Reload settings' if desired

        self._refresh_texts()
        self._update_buttons()
        self.update_permission()

        self.worker_thread = threading.Thread(target=self._render_loop, daemon=True)
        self.worker_thread.start()
        self._request_render()

    # ------------------------------------------------------------------
    # Persistence (workspace save / restore) & settings JSON
    # ------------------------------------------------------------------
    @property
    def settings_json(self) -> str:
        """All alignment settings as a JSON string (persisted with the workspace)."""
        try:
            return json.dumps(self.get_settings())
        except Exception:  # noqa: BLE001
            return ""

    @settings_json.setter
    def settings_json(self, value: str) -> None:
        # Keep saved in _saved_settings so user can reload if desired, but do not apply by default
        self._saved_settings = value or ""

    def save_current_settings(self) -> None:
        """Save current alignment parameters into memory preset."""
        self._saved_settings = self.settings_json
        logger.success("OverlayAligner: current settings saved to memory.")
        self._set_text("settings_status", "Settings saved.")

    def reload_saved_settings(self) -> None:
        """Reload saved alignment parameters from memory preset."""
        if self._saved_settings:
            try:
                self.apply_settings_json(self._saved_settings)
                logger.success("OverlayAligner: saved settings reloaded.")
                self._set_text("settings_status", "Settings reloaded.")
            except Exception as e:  # noqa: BLE001
                logger.error(f"OverlayAligner: could not reload settings: {e}")
                self._set_text("settings_status", f"Reload failed: {e}")
        else:
            logger.warning("OverlayAligner: no saved settings to reload.")
            self._set_text("settings_status", "No saved settings found.")

    def get_settings(self) -> dict[str, Any]:
        """Return the alignment settings as a JSON-compatible dict."""
        with self._lock:
            return eng.settings_to_dict(self.layers, self.g)

    def apply_settings_json(self, text: str) -> list[str]:
        """Apply settings from a JSON string."""
        return self._apply_settings(json.loads(text))

    def export_settings(self, path: str) -> None:
        """Write the alignment settings to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_settings(), f, indent=2)
        logger.success(f"OverlayAligner: settings exported to {path}")

    def import_settings(self, path: str) -> list[str]:
        """Load alignment settings from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        warnings = self._apply_settings(data)
        logger.success(f"OverlayAligner: settings imported from {path}")
        return warnings

    def _apply_settings(self, data: dict[str, Any]) -> list[str]:
        with self._lock:
            warnings = eng.apply_settings_dict(data, self.layers, self.g)
        self._fit_pending = True
        if self._ui_ready:
            self._sync_ui()
            self._request_render()
        for w in warnings:
            logger.warning(f"OverlayAligner: {w}")
        return warnings

    def update_permission(self) -> None:
        """Lock window closing in user mode."""
        from core.app_state import app_state

        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=(app_state.mode == "user"))

    # ------------------------------------------------------------------
    # Tag helpers
    # ------------------------------------------------------------------
    def _t(self, name: str) -> str:
        return f"oa_{name}_{self.UUID}"

    def _pt(self, scope: str, key: str) -> str:
        return f"oa_{scope}_{key}_{self.UUID}"

    def _params(self, scope: str) -> dict[str, Any]:
        return self.g if scope == "G" else self.layers[scope].params

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _make_themes(self) -> None:
        def _theme(c: tuple[int, int, int], h: tuple[int, int, int]) -> Any:
            with dpg.theme() as th:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, c)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, h)
            return th

        self._theme_red = _theme((120, 40, 40), (170, 60, 60))
        self._theme_green = _theme((40, 120, 40), (60, 170, 60))
        self._theme_yellow = _theme((150, 150, 40), (200, 200, 60))

    def _build_ui(self) -> None:
        self._make_themes()
        dpg.add_texture_registry(tag=self._tex_reg, show=False)
        dpg.add_raw_texture(
            width=2, height=2, default_value=np.zeros(2 * 2 * 4, np.float32),
            format=dpg.mvFormat_Float_rgba, tag=self._tex, parent=self._tex_reg,
        )

        with dpg.window(
            label=self.label, width=self.win_width, height=self.win_height,
            pos=self.pos, tag=self.winID, show=self.visible,
        ):
            with dpg.group(horizontal=True):
                with dpg.child_window(width=self._panel_w, border=False, tag=self._t("panel")):
                    self._build_top_bar()
                    with dpg.tab_bar():
                        with dpg.tab(label="A"):
                            self._build_layer_tab("A")
                        with dpg.tab(label="B"):
                            self._build_layer_tab("B")
                        with dpg.tab(label="Fit"):
                            self._build_points_tab()
                        with dpg.tab(label="Views"):
                            self._build_view_tab()
                        with dpg.tab(label="Export"):
                            self._build_export_tab()
                with dpg.child_window(border=False, width=-1, height=-1, tag=self._t("viewer_col")):
                    self._build_viewer_bar()
                    self._build_plot()

        with dpg.handler_registry(tag=self._t("handlers")):
            dpg.add_mouse_move_handler(callback=self._on_mouse_move)
            dpg.add_mouse_click_handler(callback=self._on_mouse_click)

    def _build_top_bar(self) -> None:
        s = display_scaling.scale
        bw = s(85)
        with dpg.group(horizontal=True):
            self._w_btn("WAIT A", "wait_A", bw, tag=self._t("btn_wait_A"))
            self._w_btn("WAIT B", "wait_B", bw, tag=self._t("btn_wait_B"))
            self._w_btn("Clear", "clear", s(65))
            self._w_btn("Swap A/B", "swap", s(85))
        with dpg.group(horizontal=True):
            self._w_btn("Load A...", "load_A", bw)
            self._w_btn("Load B...", "load_B", bw)
            self._w_btn("Save settings", "save_settings", s(105), tooltip="Save current alignment settings.")
            self._w_btn("Reload settings", "reload_settings", s(115), tooltip="Reload saved alignment settings.")
        dpg.add_text("", tag=self._t("status_A"), wrap=self._panel_w - s(30), color=self.A_COLOR[:3])
        dpg.add_text("", tag=self._t("status_B"), wrap=self._panel_w - s(30), color=self.B_COLOR[:3])
        dpg.add_separator()

    def _build_viewer_bar(self) -> None:
        s = display_scaling.scale
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="ROI", default_value=False, tag=self._t("roi_enable"), callback=self._on_roi_toggle)
            with dpg.tooltip(parent=self._t("roi_enable")):
                dpg.add_text("Enable ROI averaging hover box and plot annotation")
            dpg.add_text("W:")
            dpg.add_drag_int(tag=self._t("roi_w"), default_value=25, min_value=1, max_value=1000, width=s(55))
            with dpg.tooltip(parent=self._t("roi_w")):
                dpg.add_text("Width of ROI averaging box in pixels")
            dpg.add_text("H:")
            dpg.add_drag_int(tag=self._t("roi_h"), default_value=25, min_value=1, max_value=1000, width=s(55))
            with dpg.tooltip(parent=self._t("roi_h")):
                dpg.add_text("Height of ROI averaging box in pixels")
            dpg.add_text("%:")
            dpg.add_drag_int(tag=self._t("roi_thresh"), default_value=100, min_value=1, max_value=100, width=s(48))
            with dpg.tooltip(parent=self._t("roi_thresh")):
                dpg.add_text("Sensitivity (% of highest pixels included in mean calculation; 100% = all)")
            dpg.add_text("Mean:", color=(150, 180, 255))
            dpg.add_text("-", tag=self._t("roi_mean"), color=(140, 220, 140))

    def _on_roi_toggle(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args, **kwargs) -> None:
        if not app_data:
            self._hide_hover_roi()

    def _hide_hover_roi(self) -> None:
        if dpg.does_item_exist(self._t("roi_rect")):
            dpg.configure_item(self._t("roi_rect"), show=False)
        if dpg.does_item_exist(self._t("roi_annot")):
            dpg.configure_item(self._t("roi_annot"), show=False)
        if dpg.does_item_exist(self._t("roi_mean")):
            dpg.set_value(self._t("roi_mean"), "-")

    def _build_layer_tab(self, L: str) -> None:
        s = display_scaling.scale
        dpg.add_spacer(height=s(4))
        self._w_check(L, "visible", "Visible", "color")
        self._w_drag(L, "opacity", "Opacity", 0.0, 1.0, 0.005, "%.2f", "color")

        dpg.add_separator()
        with dpg.collapsing_header(label="Geometry", default_open=True):
            self._w_drag(L, "scale", "Zoom", 0.01, 50.0, 0.0005, "%.4f", "geom")
            self._w_drag(L, "aspect", "Aspect X/Y", 0.5, 2.0, 0.0005, "%.4f", "geom")
            self._w_drag(L, "rot", "Rotation (deg)", -180.0, 180.0, 0.01, "%.3f", "geom")
            self._w_drag(L, "shift_x", "Shift X (px)", -1e6, 1e6, 0.25, "%.2f", "geom")
            self._w_drag(L, "shift_y", "Shift Y (px)", -1e6, 1e6, 0.25, "%.2f", "geom")
            self._w_drag(L, "k1", "Radial k1", -1.5, 1.5, 0.0004, "%.4f", "geom")
            self._w_drag(L, "k2", "Radial k2", -1.5, 1.5, 0.0004, "%.4f", "geom")
            with dpg.group(horizontal=True):
                self._w_check(L, "flip_h", "Flip H", "geom")
                self._w_check(L, "flip_v", "Flip V", "geom")
            self._w_btn("Reset", f"reset_geom:{L}", s(90))
            dpg.add_text("", tag=self._t(f"fit_info_{L}"), wrap=self._panel_w - s(30), color=(160, 160, 160))

        dpg.add_separator()
        with dpg.collapsing_header(label="Mono & Colorization", default_open=True):
            self._w_combo(
                L, "mono_channel", "Mono channel", eng.MONO_CHANNELS, "color",
                tooltip=(
                    "For color images: select which channel to extract as monochrome signal.\n"
                    "• Gray: Standard luminance conversion.\n"
                    "• Red / Green / Blue: Specific color channel (useful e.g. Green channel for leaf tissue).\n"
                    "Grayscale images automatically use their single channel."
                ),
            )
            self._w_check(
                L, "colorize", "Colorize (apply palette; uncheck to keep natural RGB)", "color",
                tooltip=(
                    "When checked: Maps the mono channel through the false-colour palette.\n"
                    "When unchecked on RGB images: Preserves the natural RGB colors for display\n"
                    "while the mono channel settings are preserved for calculations and masking."
                ),
            )
            self._w_combo(L, "palette", "Palette", eng.PALETTES, "color")
            self._w_check(L, "invert", "Invert", "color")
            self._w_drag(L, "vmin", "Min", -1e9, 1e9, 1.0, "%.1f", "color")
            self._w_drag(L, "vmax", "Max", -1e9, 1e9, 1.0, "%.1f", "color")
            self._w_btn("Auto", f"auto_range:{L}", s(80))
            with dpg.group(horizontal=True):
                self._w_check(L, "cut_low_on", "Transparent below", "color")
                self._w_drag(L, "cut_low", "##cut_low", -1e9, 1e9, 1.0, "%.1f", "color", width=s(80))
            with dpg.group(horizontal=True):
                self._w_check(L, "cut_high_on", "Transparent above", "color")
                self._w_drag(L, "cut_high", "##cut_high", -1e9, 1e9, 1.0, "%.1f", "color", width=s(80))

    def _build_points_tab(self) -> None:
        s = display_scaling.scale
        dpg.add_spacer(height=s(4))
        with dpg.group(horizontal=True):
            dpg.add_text("View:")
            self._w_btn("A", "view:A raw", s(50), tooltip="Switch to Layer A view.")
            self._w_btn("B", "view:B raw", s(50), tooltip="Switch to Layer B view.")
            self._w_btn("Both", "view:Overlay", s(60), tooltip="Switch to Overlay canvas view (both layers).")
            dpg.add_spacer(width=s(10))
            self._w_check(
                "G", "show_points", "Show points", "overlay",
                tooltip="Show picked points (crosshairs, labels A1, B1...) and pair connection lines on the plot.",
            )
        dpg.add_separator()
        self._w_combo("G", "fit_model", "Model", eng.FIT_MODELS, "none")
        self._w_combo("G", "fit_dir", "Move layer", eng.FIT_DIRECTIONS, "none")
        self._w_btn("Fit", "fit", s(120), tooltip="Compute least-squares alignment from corresponding point pairs.")
        dpg.add_text("", tag=self._t("fit_result"), wrap=self._panel_w - s(30), color=(140, 220, 140))

        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_text("Pairs", color=(150, 180, 255))
            dpg.add_text("", tag=self._t("points_count"), color=(140, 200, 140))
        with dpg.group(horizontal=True):
            self._w_btn("Undo A", "undo:A", s(70))
            self._w_btn("Undo B", "undo:B", s(70))
            self._w_btn("Clear All", "clear_points", s(80), tooltip="Clear all picked points on both layers.")
        dpg.add_text("", tag=self._t("point_selection_info"), color=(255, 220, 80), wrap=self._panel_w - s(30))

        with dpg.child_window(height=s(200), width=self._panel_w - s(20), border=True):
            with dpg.table(
                tag=self._t("table_points"),
                header_row=True,
                borders_innerH=True,
                borders_outerH=True,
                borders_innerV=True,
                borders_outerV=True,
                row_background=True,
                scrollY=True,
                width=-1,
                height=-1,
            ):
                dpg.add_table_column(label="#", width_fixed=True, init_width_or_weight=s(26))
                dpg.add_table_column(label="Point A", width_stretch=True)
                dpg.add_table_column(label="Point B", width_stretch=True)
                dpg.add_table_column(label="x", width_fixed=True, init_width_or_weight=s(28))

    def _refresh_points_table(self) -> None:
        table_tag = self._t("table_points")
        if not dpg.does_item_exist(table_tag):
            return

        for child in (dpg.get_item_children(table_tag, 1) or []):
            try:
                dpg.delete_item(child)
            except Exception:
                pass

        with self._lock:
            pts_a = [list(p) for p in self.layers["A"].points]
            pts_b = [list(p) for p in self.layers["B"].points]
            sel = self._selected_point

        n = max(len(pts_a), len(pts_b))

        for i in range(n):
            pa = pts_a[i] if i < len(pts_a) else None
            pb = pts_b[i] if i < len(pts_b) else None

            with dpg.table_row(parent=table_tag):
                dpg.add_text(f"{i + 1}")

                # Cell A
                if pa is not None:
                    is_sel_a = (sel == ("A", i))
                    label_a = f"{'* ' if is_sel_a else ''}({pa[0]:.1f}, {pa[1]:.1f})"
                    btn_a = dpg.add_button(
                        label=label_a,
                        callback=self._on_point_cell_clicked,
                        user_data=("A", i),
                        width=-1,
                    )
                    if is_sel_a:
                        dpg.bind_item_theme(btn_a, self._theme_yellow)
                else:
                    dpg.add_text("-", color=(100, 100, 100))

                # Cell B
                if pb is not None:
                    is_sel_b = (sel == ("B", i))
                    label_b = f"{'* ' if is_sel_b else ''}({pb[0]:.1f}, {pb[1]:.1f})"
                    btn_b = dpg.add_button(
                        label=label_b,
                        callback=self._on_point_cell_clicked,
                        user_data=("B", i),
                        width=-1,
                    )
                    if is_sel_b:
                        dpg.bind_item_theme(btn_b, self._theme_yellow)
                else:
                    dpg.add_text("-", color=(100, 100, 100))

                # Delete button
                dpg.add_button(
                    label="x",
                    callback=self._on_point_delete_clicked,
                    user_data=i,
                    width=-1,
                )

    def _build_view_tab(self) -> None:
        s = display_scaling.scale
        dpg.add_spacer(height=s(4))
        self._w_combo("G", "view_mode", "View", eng.VIEW_MODES, "view")
        self._w_combo(
            "G", "blend", "Blend B over A", eng.BLEND_MODES, "color",
            tooltip=(
                "Blend modes for superimposing Layer B over Layer A:\n\n"
                "• Normal: Standard alpha overlay. B is drawn over A using B's opacity.\n"
                "• Multiply: A x B. Darkens common areas; useful to see mutual signals & masks.\n"
                "• Screen: 1 - (1-A)(1-B). Brightens and merges lighting from both images.\n"
                "• Difference: |A - B|. Absolute subtraction. Matching aligned features cancel out to black,\n"
                "  making it the ideal mode to visually verify fine alignment!"
            ),
        )
        self._w_check(
            "G", "processed_mask", "Processed as mask", "color",
            tooltip=(
                "Modulate the blend weight of Layer B over Layer A using the processed mono channel\n"
                "(selected channel, vmin/vmax range, invert, and transparency cuts).\n\n"
                "Allows displaying a natural RGB image (e.g. leaf in natural colors) while using\n"
                "its mono channel (e.g. Green channel) to define where and how Layer B (e.g. fluo) is superimposed."
            ),
        )
        self._w_combo(
            "G", "mask_source", "Mask source", eng.MASK_SOURCES, "color",
            tooltip=(
                "Which layer's processed mono channel defines the blend mask:\n\n"
                "• Layer A: Layer A's mono channel masks Layer B (e.g. fluo only over leaf tissue).\n"
                "• Layer B: Layer B's own mono channel weights its blending.\n"
                "• Both (A * B): Product of both layer masks."
            ),
        )
        self._w_combo(
            "G", "transfer_alpha", "Transfer Alpha", eng.ALPHA_TRANSFERS, "color",
            tooltip=(
                "Apply the alpha / segmentation mask of one image to the other:\n\n"
                "• None: Each layer uses its own alpha.\n"
                "• Apply Alpha A -> B: Layer A's alpha mask (e.g. from SAM or RGBA) masks Layer B.\n"
                "  If Layer B has 'Transparent below' or other cuts enabled, both masks combine (intersect).\n"
                "• Apply Alpha B -> A: Layer B's alpha mask masks Layer A.\n"
                "• Both (A * B): Both layers mask each other."
            ),
        )
        self._w_combo(
            "G", "background", "Background", eng.BACKGROUNDS, "color",
            tooltip="Canvas background fill for transparent areas.",
        )
        with dpg.group(tag=self._t("group_bg_custom"), show=(self.g.get("background") == "Custom")):
            c = self.g.get("bg_custom", [0.0, 0.0, 0.0])
            init_c = [float(c[0]) * 255.0, float(c[1]) * 255.0, float(c[2]) * 255.0, 255.0]
            dpg.add_color_edit(
                label="Custom color",
                tag=self._pt("G", "bg_custom"),
                default_value=init_c,
                no_alpha=True,
                width=self._ctl_w,
                callback=self._on_param,
                user_data=("G", "bg_custom", "color"),
            )
        dpg.add_separator()
        dpg.add_text("Intensity Scale Bar", color=(150, 180, 255))
        self._w_check(
            "G", "show_scale", "Show scale bar", "color",
            tooltip="Display a vertical color and intensity scale bar on the output.",
        )
        self._w_combo(
            "G", "scale_pos", "Scale position", eng._ALLOWED_STRINGS["scale_pos"], "color",
            tooltip="Position of the scale bar on the canvas.",
        )
        self._w_combo(
            "G", "scale_source", "Scale source", eng._ALLOWED_STRINGS["scale_source"], "color",
            tooltip=(
                "Which values define the scale bar gradient and range:\n\n"
                "• Layer B: Uses Layer B's palette and vmin/vmax range (typical for fluo).\n"
                "• Layer A: Uses Layer A's palette and vmin/vmax range.\n"
                "• Custom: Uses Custom Min / Max values."
            ),
        )
        self._w_float(
            "G", "scale_mult", "Scale multiplier", 1e-6, 1e6, "color",
            tooltip=(
                "Multiply pixel values by this factor for the displayed labels.\n"
                "Example: Set to 0.001 so raw 0–1000 pixels display as 0.000–1.000 (e.g. Fv/Fm decimal values)."
            ),
        )
        with dpg.group(horizontal=True):
            self._w_drag("G", "scale_custom_min", "Min", -1e9, 1e9, 0.01, "%.3f", "color", width=s(70), tooltip="Custom minimum value (when Scale source is Custom).")
            self._w_drag("G", "scale_custom_max", "Max", -1e9, 1e9, 0.01, "%.3f", "color", width=s(70), tooltip="Custom maximum value (when Scale source is Custom).")
        dpg.add_separator()
        self._w_check("G", "show_frames", "Show layer frames (zoom / rotation / shift / distortion)", "overlay")
        self._w_check("G", "show_grid", "Show grid", "overlay")
        self._w_int("G", "grid_div", "Grid divisions", 1, 100, "overlay")
        dpg.add_separator()
        dpg.add_text("Canvas (output area)", color=(150, 180, 255))
        self._w_combo("G", "canvas_mode", "Canvas", eng.CANVAS_MODES, "canvas")
        self._w_int("G", "canvas_w", "Custom width", 1, eng.MAX_CANVAS, "canvas")
        self._w_int("G", "canvas_h", "Custom height", 1, eng.MAX_CANVAS, "canvas")
        self._w_int("G", "preview_max", "Preview max px", 256, 8192, "geom")
        dpg.add_text(
            "A / B: native size of that layer. Union: bounding box of both transformed frames. "
            "Areas outside an image stay black and transparent.",
            wrap=self._panel_w - s(30), color=(150, 150, 150),
        )
        self._w_btn("Fit view", "fit_view", s(110))

    def _build_export_tab(self) -> None:
        s = display_scaling.scale
        dpg.add_spacer(height=s(4))
        self._w_check(
            "G", "auto_emit", "Auto emit", "export",
            tooltip="Automatically emit the visible frame whenever it updates.",
        )
        self._w_check(
            "G", "export_background", "Background", "export",
            tooltip="Composite output over the active view background instead of transparency.",
        )
        dpg.add_text("", tag=self._t("export_status"), wrap=self._panel_w - s(30), color=(140, 200, 140))
        dpg.add_separator()
        dpg.add_text("Alignment settings", color=(150, 180, 255))
        with dpg.group(horizontal=True):
            self._w_btn("Save settings", "save_settings", s(120), tooltip="Save current alignment settings.")
            self._w_btn("Reload settings", "reload_settings", s(130), tooltip="Reload saved alignment settings.")
        with dpg.group(horizontal=True):
            self._w_btn("Export settings...", "export_json", s(140), tooltip="Export settings to a JSON file.")
            self._w_btn("Import settings...", "import_json", s(140), tooltip="Import settings from a JSON file.")
        dpg.add_text("", tag=self._t("settings_status"), wrap=self._panel_w - s(30), color=(140, 200, 140))

    def _build_plot(self) -> None:
        with dpg.plot(tag=self._plot, no_menus=True, no_title=True, width=-1, height=-1, equal_aspects=True):
            dpg.add_plot_axis(
                dpg.mvXAxis, no_gridlines=True, no_tick_labels=True, no_tick_marks=True, tag=self._xaxis,
            )
            dpg.add_plot_axis(
                dpg.mvYAxis, no_gridlines=True, no_tick_labels=True, no_tick_marks=True, tag=self._yaxis,
            )
            dpg.add_image_series(self._tex, [0, 0], [2, 2], tag=self._series, parent=self._yaxis)
            dpg.draw_rectangle(
                pmin=[0, 0], pmax=[0, 0], color=(0, 255, 0, 255), thickness=2,
                tag=self._t("roi_rect"), parent=self._plot, show=False,
            )
            dpg.add_plot_annotation(
                label="", default_value=[0, 0], color=(0, 255, 0, 255), offset=(10, -10),
                tag=self._t("roi_annot"), parent=self._plot, show=False,
            )

    # ── widget helpers ───────────────────────────────────────────────
    def _w_btn(
        self, label: str, action: str, width: int | None = None, tag: str | None = None, tooltip: str | None = None,
    ) -> Any:
        kw: dict[str, Any] = {}
        if tag:
            kw["tag"] = tag
        if width:
            kw["width"] = width
        item = dpg.add_button(label=label, callback=self._on_action, user_data=action, **kw)
        if tooltip:
            with dpg.tooltip(item):
                dpg.add_text(tooltip)
        return item

    def _w_drag(
        self, scope: str, key: str, label: str, lo: float, hi: float, speed: float,
        fmt: str, kind: str, width: int | None = None, tooltip: str | None = None,
    ) -> Any:
        item = dpg.add_drag_float(
            label=label, tag=self._pt(scope, key), default_value=float(self._params(scope)[key]),
            min_value=lo, max_value=hi, speed=speed, format=fmt, width=width or self._ctl_w,
            callback=self._on_param, user_data=(scope, key, kind),
        )
        if tooltip:
            with dpg.tooltip(item):
                dpg.add_text(tooltip)
        return item

    def _w_check(self, scope: str, key: str, label: str, kind: str, tooltip: str | None = None) -> Any:
        item = dpg.add_checkbox(
            label=label, tag=self._pt(scope, key), default_value=bool(self._params(scope)[key]),
            callback=self._on_param, user_data=(scope, key, kind),
        )
        if tooltip:
            with dpg.tooltip(item):
                dpg.add_text(tooltip)
        return item

    def _w_combo(self, scope: str, key: str, label: str, items: list[str], kind: str, tooltip: str | None = None) -> Any:
        item = dpg.add_combo(
            items=items, label=label, tag=self._pt(scope, key), default_value=str(self._params(scope)[key]),
            width=self._ctl_w, callback=self._on_param, user_data=(scope, key, kind),
        )
        if tooltip:
            with dpg.tooltip(item):
                dpg.add_text(tooltip)
        return item

    def _w_int(self, scope: str, key: str, label: str, lo: int, hi: int, kind: str, tooltip: str | None = None) -> Any:
        item = dpg.add_input_int(
            label=label, tag=self._pt(scope, key), default_value=int(self._params(scope)[key]),
            min_value=lo, max_value=hi, min_clamped=True, max_clamped=True, step=0,
            width=self._ctl_w, callback=self._on_param, user_data=(scope, key, kind),
        )
        if tooltip:
            with dpg.tooltip(item):
                dpg.add_text(tooltip)
        return item

    def _w_float(self, scope: str, key: str, label: str, lo: float, hi: float, kind: str, tooltip: str | None = None) -> Any:
        item = dpg.add_input_float(
            label=label, tag=self._pt(scope, key), default_value=float(self._params(scope)[key]),
            min_value=lo, max_value=hi, min_clamped=True, max_clamped=True, step=0, format="%.3f",
            width=self._ctl_w, callback=self._on_param, user_data=(scope, key, kind),
        )
        if tooltip:
            with dpg.tooltip(item):
                dpg.add_text(tooltip)
        return item

    # ------------------------------------------------------------------
    # UI <-> state synchronisation
    # ------------------------------------------------------------------
    def _sync_ui(self) -> None:
        """Push every state value to its widget."""
        for scope in ("A", "B", "G"):
            with self._lock:
                items = list(self._params(scope).items())
            for key, val in items:
                tag = self._pt(scope, key)
                if dpg.does_item_exist(tag):
                    try:
                        if scope == "G" and key == "bg_custom":
                            dpg.set_value(tag, [float(val[0]) * 255.0, float(val[1]) * 255.0, float(val[2]) * 255.0, 255.0])
                        else:
                            dpg.set_value(tag, val)
                    except Exception:  # noqa: BLE001
                        pass
        if dpg.does_item_exist(self._t("group_bg_custom")):
            bg_mode = self.g.get("background", "Checker")
            dpg.configure_item(self._t("group_bg_custom"), show=(bg_mode == "Custom"))
        self._update_range_speeds()
        self._refresh_texts()
        self._update_overlays()

    def _set_param(self, scope: str, key: str, value: Any) -> None:
        with self._lock:
            self._params(scope)[key] = value
        tag = self._pt(scope, key)
        if dpg.does_item_exist(tag):
            if scope == "G" and key == "bg_custom":
                try:
                    c = [float(x) for x in value[:3]]
                    dpg.set_value(tag, [c[0] * 255.0, c[1] * 255.0, c[2] * 255.0, 255.0])
                except Exception:
                    pass
            else:
                dpg.set_value(tag, value)
        if scope == "G" and key == "background" and dpg.does_item_exist(self._t("group_bg_custom")):
            dpg.configure_item(self._t("group_bg_custom"), show=(value == "Custom"))

    def _update_range_speeds(self) -> None:
        for L in ("A", "B"):
            with self._lock:
                li = self.layers[L].image
                p = self.layers[L].params
                span = abs(float(p["vmax"]) - float(p["vmin"]))
                if li is not None:
                    span = max(span, abs(li.dmax - li.dmin))
            speed = max(span / 400.0, 0.01)
            for key in ("vmin", "vmax", "cut_low", "cut_high"):
                tag = self._pt(L, key)
                if dpg.does_item_exist(tag):
                    dpg.configure_item(tag, speed=speed)

    def _refresh_texts(self) -> None:
        with self._lock:
            lines = {}
            fit_lines = {}
            for L, layer in self.layers.items():
                li = layer.image
                if li is None:
                    lines[L] = f"{L}: (no image)"
                elif li.kind == "gray":
                    lines[L] = f"{L}: {li.w}x{li.h} {li.desc}  range [{li.dmin:g} .. {li.dmax:g}]"
                else:
                    lines[L] = f"{L}: {li.w}x{li.h} {li.desc}"
                fit_lines[L] = "Point fit active." \
                    if layer.has_fit() else ""
            na, nb = len(self.layers["A"].points), len(self.layers["B"].points)
            sel = self._selected_point
            if sel is not None:
                L_sel, idx_sel = sel
                pts_list = self.layers[L_sel].points
                if 0 <= idx_sel < len(pts_list):
                    pt = pts_list[idx_sel]
                    sel_text = f"Selected: Point {L_sel}#{idx_sel + 1} ({pt[0]:.1f}, {pt[1]:.1f}) - Click image in {L_sel} to replace"
                else:
                    sel_text = ""
                    self._selected_point = None
            else:
                sel_text = ""
        for L in ("A", "B"):
            if dpg.does_item_exist(self._t(f"status_{L}")):
                dpg.set_value(self._t(f"status_{L}"), lines[L])
            if dpg.does_item_exist(self._t(f"fit_info_{L}")):
                dpg.set_value(self._t(f"fit_info_{L}"), fit_lines[L])
        if dpg.does_item_exist(self._t("points_count")):
            dpg.set_value(self._t("points_count"), f"A: {na} | B: {nb}")
        if dpg.does_item_exist(self._t("point_selection_info")):
            dpg.set_value(self._t("point_selection_info"), sel_text)
        self._refresh_points_table()

    def _update_buttons(self) -> None:
        for L in ("A", "B"):
            tag = self._t(f"btn_wait_{L}")
            if not dpg.does_item_exist(tag):
                continue
            if self._armed == L:
                dpg.bind_item_theme(tag, self._theme_yellow)
            elif self.layers[L].image is not None:
                dpg.bind_item_theme(tag, self._theme_green)
            else:
                dpg.bind_item_theme(tag, self._theme_red)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_point_cell_clicked(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        L, idx = user_data
        with self._lock:
            if self._selected_point == (L, idx):
                self._selected_point = None
            else:
                self._selected_point = (L, idx)

        if self._selected_point is not None:
            raw_view = f"{L} raw"
            if self.g["view_mode"] != raw_view:
                self._set_param("G", "view_mode", raw_view)
                self._fit_pending = True
                self._request_render()

        self._refresh_texts()
        self._update_overlays()

    def _on_point_delete_clicked(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        i = int(user_data)
        with self._lock:
            for L in ("A", "B"):
                pts = self.layers[L].points
                if i < len(pts):
                    pts.pop(i)
            if self._selected_point is not None:
                if self._selected_point[1] == i:
                    self._selected_point = None
                elif self._selected_point[1] > i:
                    self._selected_point = (self._selected_point[0], self._selected_point[1] - 1)
        self._refresh_texts()
        self._update_overlays()

    def _on_param(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        scope, key, kind = user_data
        with self._lock:
            if scope == "G":
                if key == "bg_custom":
                    try:
                        vals = [float(c) for c in app_data[:3]]
                        if max(vals) > 1.0:
                            vals = [c / 255.0 for c in vals]
                        self.g[key] = vals
                    except Exception:
                        self.g[key] = [0.0, 0.0, 0.0]
                else:
                    self.g[key] = app_data
            else:
                self.layers[scope].params[key] = app_data
                if key in ("vmin", "vmax"):
                    self.layers[scope].range_user_set = True

        if scope == "G" and key == "background":
            if dpg.does_item_exist(self._t("group_bg_custom")):
                dpg.configure_item(self._t("group_bg_custom"), show=(app_data == "Custom"))

        if kind == "canvas" or kind == "view":
            self._fit_pending = True
        if kind == "overlay":
            self._update_overlays()
        if kind in ("geom", "canvas", "view", "color", "export"):
            self._request_render()

    def _on_action(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        action = str(user_data)
        parts = action.split(":")
        cmd = parts[0] if parts[0] not in ("wait_A", "wait_B", "load_A", "load_B") else action
        try:
            if cmd in ("wait_A", "wait_B"):
                L = cmd[-1]
                self._armed = None if self._armed == L else L
                self._update_buttons()
            elif cmd in ("load_A", "load_B"):
                self._pick_image_file(cmd[-1])
            elif cmd == "clear":
                self.clear_frames()
            elif cmd == "swap":
                self._swap_layers()
            elif cmd == "reset_geom":
                self._reset_geometry(parts[1])
            elif cmd == "clear_fit":
                with self._lock:
                    self.layers[parts[1]].fit_H = np.eye(3)
                self._after_geometry_change()
            elif cmd == "save_settings":
                self.save_current_settings()
            elif cmd == "reload_settings":
                self.reload_saved_settings()
            elif cmd == "auto_range":
                self._auto_range(parts[1])
            elif cmd == "view":
                self._set_param("G", "view_mode", parts[1])
                self._fit_pending = True
                self._update_overlays()
                self._request_render()
            elif cmd == "undo":
                with self._lock:
                    if self.layers[parts[1]].points:
                        self.layers[parts[1]].points.pop()
                self._refresh_texts()
                self._update_overlays()
            elif cmd == "clear_points":
                with self._lock:
                    for layer in self.layers.values():
                        layer.points.clear()
                self._refresh_texts()
                self._update_overlays()
            elif cmd == "fit":
                self._do_fit()
            elif cmd == "fit_view":
                self._fit_axes()
            elif cmd == "emit":
                self._emit_async()
            elif cmd == "save_png":
                self._pick_png_path()
            elif cmd == "export_json":
                self._pick_json_export()
            elif cmd == "import_json":
                self._pick_json_import()
        except Exception as e:  # noqa: BLE001
            logger.error(f"OverlayAligner action '{action}' failed: {e}")

    def _on_mouse_move(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        try:
            if not dpg.does_item_exist(self._plot) or not dpg.is_item_hovered(self._plot):
                self._hide_hover_roi()
                return

            roi_enabled = False
            if dpg.does_item_exist(self._t("roi_enable")):
                roi_enabled = bool(dpg.get_value(self._t("roi_enable")))

            if not roi_enabled:
                self._hide_hover_roi()
                return

            mx, my = dpg.get_plot_mouse_pos()
            mode = self.g["view_mode"]
            roi_val_text = ""
            annot_text = ""

            if mode == "Overlay":
                ox, oy, W, H = self._last_canvas
                pw, ph = W, H
            else:
                L = mode[0]
                with self._lock:
                    li = self.layers[L].image
                if li is None:
                    self._hide_hover_roi()
                    return
                pw, ph = li.w, li.h

            rw = int(dpg.get_value(self._t("roi_w")) or 25)
            rh = int(dpg.get_value(self._t("roi_h")) or 25)
            thresh_pct = float(dpg.get_value(self._t("roi_thresh")) or 100.0)

            rx1 = mx - rw / 2.0
            rx2 = mx + rw / 2.0
            ry1 = my - rh / 2.0
            ry2 = my + rh / 2.0

            ix1 = max(0, int(round(rx1)))
            ix2 = min(pw, int(round(rx2)))
            iy1 = max(0, int(round(ph - ry2)))
            iy2 = min(ph, int(round(ph - ry1)))

            if ix2 <= ix1 or iy2 <= iy1:
                self._hide_hover_roi()
                return

            if mode == "Overlay":
                with self._lock:
                    bgr_img = self._current_rendered_bgr
                    a_img = self._current_rendered_a
                if bgr_img is not None and bgr_img.shape[0] >= iy2 and bgr_img.shape[1] >= ix2:
                    patch = (bgr_img[iy1:iy2, ix1:ix2, :3] * 255.0).astype(np.float32)
                    if patch.size > 0:
                        gray = cv2.cvtColor(np.clip(patch, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                        if a_img is not None and a_img.shape[0] >= iy2 and a_img.shape[1] >= ix2:
                            mask_alpha = a_img[iy1:iy2, ix1:ix2] > 0.05
                        else:
                            mask_alpha = np.ones(gray.shape, dtype=bool)

                        if thresh_pct < 100.0:
                            vals_base = gray[mask_alpha] if np.any(mask_alpha) else gray
                            p_thresh = np.percentile(vals_base, 100.0 - thresh_pct)
                            mask = (gray >= p_thresh) & mask_alpha
                        else:
                            mask = mask_alpha

                        val_px = patch[mask] if np.any(mask) else patch.reshape(-1, 3)
                        if val_px.size > 0:
                            b_m, g_m, r_m = np.mean(val_px, axis=0)
                            roi_val_text = f"RGB: {r_m:.1f}, {g_m:.1f}, {b_m:.1f}"
                            annot_text = f"RGB: {r_m:.0f}, {g_m:.0f}, {b_m:.0f}"
            else:
                L = mode[0]
                with self._lock:
                    li = self.layers[L].image
                if li is not None and li.h >= iy2 and li.w >= ix2:
                    if li.kind == "gray":
                        val_arr = li.stack[:, :, 0] / np.maximum(li.stack[:, :, 1], 1e-6)
                        cov = li.stack[iy1:iy2, ix1:ix2, 1]
                        patch = val_arr[iy1:iy2, ix1:ix2]
                        mask_cov = cov > 0.05
                        patch_valid = patch[mask_cov] if np.any(mask_cov) else patch.ravel()
                        if patch_valid.size > 0:
                            if thresh_pct < 100.0:
                                p_thresh = np.percentile(patch_valid, 100.0 - thresh_pct)
                                valid = patch_valid[patch_valid >= p_thresh]
                            else:
                                valid = patch_valid
                            if valid.size > 0:
                                m_val = float(np.mean(valid))
                                s_val = float(np.std(valid))
                                roi_val_text = f"{m_val:.2f} ± {s_val:.2f}"
                                annot_text = f"{m_val:.1f} , {s_val:.1f}"
                    else:
                        patch = li.stack[iy1:iy2, ix1:ix2, :3].astype(np.float32)
                        if patch.size > 0:
                            gray = cv2.cvtColor(np.clip(patch, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                            if thresh_pct < 100.0:
                                p_thresh = np.percentile(gray, 100.0 - thresh_pct)
                                mask = gray >= p_thresh
                            else:
                                mask = np.ones(gray.shape, dtype=bool)
                            val_px = patch[mask] if np.any(mask) else patch.reshape(-1, 3)
                            if val_px.size > 0:
                                b_m, g_m, r_m = np.mean(val_px, axis=0)
                                roi_val_text = f"RGB: {r_m:.1f}, {g_m:.1f}, {b_m:.1f}"
                                annot_text = f"RGB: {r_m:.0f}, {g_m:.0f}, {b_m:.0f}"

            if roi_val_text:
                dpg.set_value(self._t("roi_mean"), roi_val_text)

            if not dpg.does_item_exist(self._t("roi_rect")):
                dpg.draw_rectangle(
                    pmin=[rx1, ry1], pmax=[rx2, ry2], color=(0, 255, 0, 255), thickness=2,
                    tag=self._t("roi_rect"), parent=self._plot, show=True,
                )
            else:
                dpg.configure_item(self._t("roi_rect"), pmin=[rx1, ry1], pmax=[rx2, ry2], show=True)

            if not dpg.does_item_exist(self._t("roi_annot")):
                dpg.add_plot_annotation(
                    label=annot_text, default_value=[rx2, ry2], color=(0, 255, 0, 255),
                    offset=(10, -10), tag=self._t("roi_annot"), parent=self._plot, show=True,
                )
            else:
                dpg.configure_item(self._t("roi_annot"), default_value=[rx2, ry2], label=annot_text, show=True)
        except Exception:  # noqa: BLE001
            pass

    def _on_mouse_click(self, sender: Any, app_data: Any, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        btn = app_data[0] if isinstance(app_data, (list, tuple)) else app_data
        ctrl = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(getattr(dpg, "mvKey_RControl", dpg.mvKey_LControl))
        shift = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(getattr(dpg, "mvKey_RShift", dpg.mvKey_LShift))
        replace_mode = (self._selected_point is not None and btn in (0, 2) and not shift)
        add = ((btn == 0 and ctrl) or btn == 2) and not replace_mode
        remove = btn == 0 and shift
        if not (add or remove or replace_mode):
            return
        try:
            if not dpg.does_item_exist(self._plot) or not dpg.is_item_hovered(self._plot):
                return
            mode = self.g["view_mode"]
            if mode == "Overlay":
                if dpg.does_item_exist(self._t("roi_mean")):
                    dpg.set_value(self._t("roi_mean"), "Switch to 'A' or 'B' view to pick/replace points.")
                return
            L = mode[0]
            mx, my = dpg.get_plot_mouse_pos()
            with self._lock:
                li = self.layers[L].image
                if li is None:
                    return
                x, y = mx - 0.5, (li.h - my) - 0.5
                pts = self.layers[L].points
                if replace_mode and self._selected_point is not None and self._selected_point[0] == L:
                    idx = self._selected_point[1]
                    if -0.5 <= x < li.w - 0.5 and -0.5 <= y < li.h - 0.5:
                        if 0 <= idx < len(pts):
                            pts[idx] = [float(x), float(y)]
                        else:
                            pts.append([float(x), float(y)])
                        self._selected_point = None
                elif remove:
                    if pts:
                        d = [(px - x) ** 2 + (py - y) ** 2 for px, py in pts]
                        pts.pop(int(np.argmin(d)))
                        self._selected_point = None
                elif add:
                    if -0.5 <= x < li.w - 0.5 and -0.5 <= y < li.h - 0.5:
                        pts.append([float(x), float(y)])
            self._refresh_texts()
            self._update_overlays()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"OverlayAligner click ignored: {e}")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------


    def _reset_geometry(self, L: str) -> None:
        with self._lock:
            self.layers[L].reset_geometry()
        self._after_geometry_change()

    def _after_geometry_change(self, sync: bool = True) -> None:
        if sync:
            self._sync_ui()
        else:
            self._update_overlays()
        self._request_render()

    def _auto_range(self, L: str) -> None:
        with self._lock:
            layer = self.layers[L]
            li = layer.image
            if li is None:
                return
            if li.kind == "gray":
                lo, hi = li.dmin, li.dmax
            else:
                bgr = li.stack[:, :, :3]
                ch = str(layer.params.get("mono_channel", "Gray"))
                if ch in ("Red", "R"):
                    arr = bgr[:, :, 2].astype(np.float32)
                elif ch in ("Green", "G"):
                    arr = bgr[:, :, 1].astype(np.float32)
                elif ch in ("Blue", "B"):
                    arr = bgr[:, :, 0].astype(np.float32)
                else:
                    arr = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2GRAY).astype(np.float32)
                lo, hi = float(arr.min()), float(arr.max())
            if hi <= lo:
                hi = lo + 1.0
            layer.params["vmin"], layer.params["vmax"] = float(lo), float(hi)
            layer.range_user_set = False
        self._sync_ui()
        self._request_render()

    def clear_frames(self) -> None:
        """Remove both images (settings and points are kept)."""
        with self._lock:
            for layer in self.layers.values():
                layer.image = None
            self._fresh = {"A": False, "B": False}
            self._armed = None
            self._render_cache.clear()
        self._fit_pending = True
        self._refresh_texts()
        self._update_buttons()
        self._update_overlays()
        self._request_render()

    def _swap_layers(self) -> None:
        with self._lock:
            a, b = self.layers["A"], self.layers["B"]
            a.image, b.image = b.image, a.image
            a.points, b.points = b.points, a.points
            for layer in (a, b):
                layer.reset_geometry()
                layer.range_user_set = False
                li = layer.image
                if li is not None:
                    lo, hi = (li.dmin, li.dmax) if li.kind == "gray" else (0.0, 255.0)
                    layer.params["vmin"], layer.params["vmax"] = lo, (hi if hi > lo else lo + 1.0)
            self._render_cache.clear()
        self._fit_pending = True
        self._sync_ui()
        self._update_buttons()
        self._request_render()

    def _do_fit(self) -> None:
        with self._lock:
            # Reset both layer geometries and fit matrices before computing fit
            for layer in self.layers.values():
                layer.reset_geometry()
                layer.fit_H = np.eye(3)

            mov_name = "B" if self.g["fit_dir"] == "B -> A" else "A"
            ref_name = "A" if mov_name == "B" else "B"
            mov, ref = self.layers[mov_name], self.layers[ref_name]
            if mov.image is None or ref.image is None:
                msg = "Load both images first."
            else:
                n = min(len(mov.points), len(ref.points))
                if n == 0:
                    msg = "Pick point pairs first (A or B view, Ctrl+Click)."
                else:
                    src = eng.src_to_pre(np.array(mov.points[:n]), mov.params, mov.image.w, mov.image.h)
                    dst = eng.src_to_world(
                        np.array(ref.points[:n]), ref.params, ref.fit_H, ref.image.w, ref.image.h
                    )
                    H, info, mean_err, max_err = eng.fit_transform(src, dst, str(self.g["fit_model"]))
                    if H is None:
                        msg = info
                    else:
                        if self.g["fit_model"] != "Homography":
                            decomp = eng.decompose_affine_transform(H, mov.image.w, mov.image.h)
                            for k, v in decomp.items():
                                mov.params[k] = v
                            mov.fit_H = np.eye(3)
                        else:
                            mov.fit_H = H
                            mov.reset_fine_geometry()

                        extra = ""
                        if len(mov.points) != len(ref.points):
                            extra = f" (used the first {n} pairs; counts differ)"
                        msg = (
                            f"{self.g['fit_model']} fit of {mov_name} onto {ref_name}: "
                            f"mean error {mean_err:.2f} px, max {max_err:.2f} px, {n} pairs{extra}."
                        )
                        self._render_cache.clear()
        self._last_fit_msg = msg
        if dpg.does_item_exist(self._t("fit_result")):
            dpg.set_value(self._t("fit_result"), msg)
        self._sync_ui()
        self._request_render()

    def _fit_axes(self) -> None:
        if dpg.does_item_exist(self._xaxis) and dpg.does_item_exist(self._yaxis):
            dpg.set_axis_limits_auto(self._xaxis)
            dpg.set_axis_limits_auto(self._yaxis)
            dpg.fit_axis_data(self._xaxis)
            dpg.fit_axis_data(self._yaxis)

    # ── file pickers ─────────────────────────────────────────────────
    def _pick_image_file(self, slot: str) -> None:
        from core.file_explorer import file_explorer

        def _cb(path: str, *a: Any, **kw: Any) -> None:
            if path:
                self._load_image_file(path, slot)

        file_explorer.select_file(default_path=self._export_folder or "", extensions=IMAGE_EXTENSIONS, callback=_cb)

    def _load_image_file(self, path: str, slot: str) -> None:
        try:
            img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        except Exception as e:  # noqa: BLE001
            logger.error(f"OverlayAligner: cannot read {path}: {e}")
            return
        if img is None:
            logger.error(f"OverlayAligner: unsupported image file {path}")
            return
        self._ingest(img, slot=slot)

    def _pick_png_path(self) -> None:
        from core.file_explorer import file_explorer

        def _cb(path: str, *a: Any, **kw: Any) -> None:
            if not path:
                return
            if not path.lower().endswith(".png"):
                path += ".png"
            threading.Thread(target=self._emit_worker, kwargs={"save_path": path, "emit": False}, daemon=True).start()

        file_explorer.save_file(
            default_path=self._export_folder or "", default_name=f"{self._group_name}_overlay.png",
            extensions=[("PNG (RGBA)", "*.png")], callback=_cb,
        )

    def _pick_json_export(self) -> None:
        from core.file_explorer import file_explorer

        def _cb(path: str, *a: Any, **kw: Any) -> None:
            if not path:
                return
            if not path.lower().endswith(".json"):
                path += ".json"
            try:
                self.export_settings(path)
                self._set_text("settings_status", f"Exported: {os.path.basename(path)}")
            except Exception as e:  # noqa: BLE001
                self._set_text("settings_status", f"Export failed: {e}")

        file_explorer.save_file(
            default_path=self._export_folder or "", default_name="overlay_alignment.json",
            extensions=[("JSON", "*.json")], callback=_cb,
        )

    def _pick_json_import(self) -> None:
        from core.file_explorer import file_explorer

        def _cb(path: str, *a: Any, **kw: Any) -> None:
            if not path:
                return
            try:
                warnings = self.import_settings(path)
                msg = f"Imported: {os.path.basename(path)}"
                if warnings:
                    msg += "\n" + "\n".join(warnings)
                self._set_text("settings_status", msg)
            except Exception as e:  # noqa: BLE001
                self._set_text("settings_status", f"Import failed: {e}")

        file_explorer.select_file(
            default_path=self._export_folder or "", extensions=[("JSON", "*.json")], callback=_cb,
        )

    def _set_text(self, name: str, text: str) -> None:
        tag = self._t(name)
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, text)

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def input_cb(self, *args: Any, **kwargs: Any) -> None:
        """Receive frames, file paths, folder / group names and commands."""
        data_type = kwargs.get("data_type")
        data = kwargs.get("data")
        if data is None:
            data = args[0] if args else None
        if data is None:
            data = kwargs.get("frame")
        if data is None:
            data = kwargs.get("sample")
        if data is None:
            return
        slot = str(kwargs.get("slot", "") or "").upper()
        slot = slot if slot in ("A", "B") else ""

        # Commands
        if isinstance(data, dict) and "frame" not in data:
            self._handle_command(data)
            return
        if isinstance(data, dict):
            self._ingest(np.asarray(data["frame"]), slot, alpha=data.get("mask"))
            return

        # Strings: folder / file / group name
        if isinstance(data, str):
            if data_type == IOTypes.TEXT:
                self._group_name = data
            elif os.path.isdir(data):
                self._export_folder = data
            elif os.path.isfile(data):
                self._load_image_file(data, slot)
            else:
                self._group_name = data
            return

        # (frame, mask) pair or (frame, name) tuple
        alpha = None
        if isinstance(data, (tuple, list)) and data and isinstance(data[0], np.ndarray):
            if len(data) == 2 and isinstance(data[1], np.ndarray):
                alpha = data[1]
            data = data[0]
        if isinstance(data, np.ndarray):
            self._ingest(data, slot, alpha=alpha)

    def _handle_command(self, cmd: dict[str, Any]) -> None:
        action = str(cmd.get("action", "")).lower()
        if cmd.get("clear") or action == "clear":
            self.clear_frames()
        elif action in ("wait_a", "wait_b"):
            self._armed = action[-1].upper()
            self._update_buttons()
        elif action == "emit":
            self._emit_async()
        elif action == "load_settings" and cmd.get("path"):
            try:
                self.import_settings(str(cmd["path"]))
            except Exception as e:  # noqa: BLE001
                logger.error(f"OverlayAligner: load_settings failed: {e}")

    def _ingest(self, frame: np.ndarray, slot: str = "", alpha: np.ndarray | None = None) -> None:
        try:
            li = eng.LayerImage(frame, alpha)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"OverlayAligner: rejected frame ({e})")
            return

        with self._lock:
            explicit = False
            if slot in ("A", "B"):
                target, explicit = slot, True
            elif self._armed:
                target, explicit = self._armed, True
            elif self.layers["A"].image is None:
                target = "A"
            elif self.layers["B"].image is None:
                target = "B"
            else:
                return  # both slots filled and nothing armed
            if self._armed == target:
                self._armed = None

            layer = self.layers[target]
            first = layer.image is None
            size_changed = first or layer.image.size != li.size  # type: ignore[union-attr]
            layer.image = li
            if first and not layer.range_user_set:
                lo, hi = (li.dmin, li.dmax) if li.kind == "gray" else (0.0, 255.0)
                layer.params["vmin"], layer.params["vmax"] = float(lo), float(hi if hi > lo else lo + 1.0)
            self._fresh[target] = True
            both = self.layers["A"].image is not None and self.layers["B"].image is not None
            trigger = both and (explicit or (self._fresh["A"] and self._fresh["B"]))
            if trigger:
                self._fresh = {"A": False, "B": False}
            if size_changed:
                self._fit_pending = True

        if dpg.does_item_exist(self._pt(target, "vmin")):
            dpg.set_value(self._pt(target, "vmin"), float(self.layers[target].params["vmin"]))
            dpg.set_value(self._pt(target, "vmax"), float(self.layers[target].params["vmax"]))
        self._update_range_speeds()
        self._refresh_texts()
        self._update_buttons()
        self._update_overlays()
        self._request_render()

    # ------------------------------------------------------------------
    # Rendering (worker thread) and presentation (main thread)
    # ------------------------------------------------------------------
    def _snapshot(self) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        with self._lock:
            snaps = {
                name: {
                    "image": layer.image,
                    "params": dict(layer.params),
                    "fit_H": layer.fit_H.copy(),
                    "points": [list(p) for p in layer.points],
                }
                for name, layer in self.layers.items()
            }
            return snaps, dict(self.g)

    def _request_render(self) -> None:
        self._dirty.set()

    def _render_loop(self) -> None:
        while self.running:
            self._dirty.wait(timeout=0.25)
            if not self.running:
                break
            if not self._dirty.is_set():
                continue
            self._dirty.clear()
            try:
                self._render_and_present()
            except Exception as e:  # noqa: BLE001
                logger.error(f"OverlayAligner render error: {e}")

    def _render_and_present(self) -> None:
        from core.automation_manager import automation_manager

        snaps, g = self._snapshot()
        mode = g["view_mode"]
        if mode in ("A raw", "B raw"):
            bgr, a, pw, ph = eng.render_raw(snaps[mode[0]], g, processed=False)
        elif mode in ("A processed", "B processed"):
            bgr, a, pw, ph = eng.render_raw(snaps[mode[0]], g, processed=True)
        else:
            res = eng.render_overlay(
                snaps, g, preview_max=int(min(max(g["preview_max"], 256), 8192)), cache=self._render_cache,
            )
            bgr, a, pw, ph = res.bgr, res.alpha, res.canvas_w, res.canvas_h
        flat = eng.to_display_rgba(bgr, a, str(g["background"]), g.get("bg_custom"))
        h, w = a.shape[:2]
        with self._lock:
            self._current_rendered_bgr = bgr
            self._current_rendered_a = a
            self._current_rendered_pw = pw
            self._current_rendered_ph = ph
            self._pending_frame = (flat, w, h, pw, ph)

        self._schedule_presentation()

        if g.get("auto_emit") and (snaps["A"]["image"] is not None or snaps["B"]["image"] is not None):
            if not self._dirty.is_set():
                self._emit_visible_frame(bgr, a, g)
            else:
                self._schedule_emit(delay=0.1)

    def _schedule_presentation(self) -> None:
        from core.automation_manager import automation_manager
        with self._lock:
            if not self._present_pending:
                self._present_pending = True
                automation_manager.post_to_main_thread(self._do_present, wait=False)

    def _do_present(self) -> None:
        with self._lock:
            frame = self._pending_frame
            self._pending_frame = None
            self._present_pending = False
        if frame is None:
            return
        flat, w, h, pw, ph = frame
        self._present(flat, w, h, pw, ph)

    def _present(self, flat: np.ndarray, w: int, h: int, pw: int, ph: int) -> None:
        """Main-thread texture upload."""
        if not self.running or not dpg.does_item_exist(self._plot):
            return
        if w != self._tex_w or h != self._tex_h or not dpg.does_item_exist(self._tex):
            if dpg.does_item_exist(self._series):
                dpg.delete_item(self._series)
            if dpg.does_item_exist(self._tex):
                dpg.delete_item(self._tex)
            dpg.add_raw_texture(
                width=w, height=h, default_value=flat, format=dpg.mvFormat_Float_rgba,
                tag=self._tex, parent=self._tex_reg,
            )
            dpg.add_image_series(self._tex, [0, 0], [pw, ph], tag=self._series, parent=self._yaxis)
            self._tex_w, self._tex_h = w, h
        else:
            dpg.set_value(self._tex, flat)
            dpg.configure_item(self._series, bounds_min=[0, 0], bounds_max=[pw, ph])
        if self._fit_pending:
            self._fit_axes()
            self._fit_pending = False
        self._update_overlays()

    # ------------------------------------------------------------------
    # Overlays (frames / grid / points) drawn in plot coordinates
    # ------------------------------------------------------------------
    def _clear_overlays(self) -> None:
        """Purge all custom overlay draw items from the plot."""
        if not dpg.does_item_exist(self._plot):
            self._overlay_items = []
            return

        roi_tags = {self._t("roi_rect"), self._t("roi_annot")}
        for item in self._overlay_items:
            try:
                if item not in roi_tags and dpg.does_item_exist(item):
                    dpg.delete_item(item)
            except Exception:
                pass
        self._overlay_items = []

    def _update_overlays(self) -> None:
        if not self._ui_ready or not dpg.does_item_exist(self._plot):
            return
        if threading.current_thread() is not threading.main_thread():
            from core.automation_manager import automation_manager
            automation_manager.post_to_main_thread(self._update_overlays, wait=False)
            return

        self._clear_overlays()
        snaps, g = self._snapshot()
        mode = g["view_mode"]
        items = self._overlay_items
        plot = self._plot
        try:
            if mode == "Overlay":
                ox, oy, W, H = eng.compute_canvas(snaps, g)
                self._last_canvas = (ox, oy, W, H)
                pw, ph = W, H

                def to_plot(xy: Any) -> list[list[float]]:
                    arr = np.asarray(xy, dtype=float).reshape(-1, 2)
                    return [[float(x - ox + 0.5), float(H - (y - oy + 0.5))] for x, y in arr]
            else:
                li = snaps[mode[0]]["image"]
                if li is None:
                    return
                pw, ph = li.w, li.h

                def to_plot(xy: Any) -> list[list[float]]:
                    arr = np.asarray(xy, dtype=float).reshape(-1, 2)
                    return [[float(x + 0.5), float(ph - (y + 0.5))] for x, y in arr]

            if pw <= 0 or ph <= 0:
                return

            items.append(dpg.draw_polyline(
                [[0, 0], [pw, 0], [pw, ph], [0, ph]], closed=True, color=self.CANVAS_COLOR, thickness=1, parent=plot,
            ))

            if g["show_grid"]:
                div = max(1, int(g["grid_div"]))
                grid_colors = {
                    "A": (80, 220, 255, 140),  # Blue grid for Layer A
                    "B": (255, 90, 200, 140),  # Pink grid for Layer B
                }
                layers_to_grid = ["A", "B"] if mode == "Overlay" else ([mode[0]] if mode[0] in ("A", "B") else [])
                for L in layers_to_grid:
                    sn = snaps[L]
                    li = sn["image"]
                    if li is None:
                        continue
                    color = grid_colors.get(L, self.GRID_COLOR)
                    w_img, h_img = li.w, li.h
                    step_x = w_img / float(div)
                    step_y = h_img / float(div)
                    n_samples = 25

                    # Vertical grid lines
                    for i in range(div + 1):
                        gx = min(i * step_x, float(w_img))
                        ys = np.linspace(0.0, float(h_img), n_samples)
                        pts = np.column_stack([np.full_like(ys, gx), ys])
                        if mode == "Overlay":
                            w_pts = eng.src_to_world(pts, sn["params"], sn["fit_H"], w_img, h_img)
                        else:
                            w_pts = pts
                        if np.all(np.isfinite(w_pts)):
                            items.append(dpg.draw_polyline(to_plot(w_pts), closed=False, color=color, thickness=1, parent=plot))

                    # Horizontal grid lines
                    for j in range(div + 1):
                        gy = min(j * step_y, float(h_img))
                        xs = np.linspace(0.0, float(w_img), n_samples)
                        pts = np.column_stack([xs, np.full_like(xs, gy)])
                        if mode == "Overlay":
                            w_pts = eng.src_to_world(pts, sn["params"], sn["fit_H"], w_img, h_img)
                        else:
                            w_pts = pts
                        if np.all(np.isfinite(w_pts)):
                            items.append(dpg.draw_polyline(to_plot(w_pts), closed=False, color=color, thickness=1, parent=plot))

            colors = {"A": self.A_COLOR, "B": self.B_COLOR}
            marker = 0.008 * max(pw, ph)

            if mode == "Overlay" and g["show_frames"]:
                for L in ("A", "B"):
                    sn = snaps[L]
                    li = sn["image"]
                    if li is None:
                        continue
                    n = 24
                    poly = eng.footprint_world(sn["params"], sn["fit_H"], li.w, li.h, n=n)
                    if not np.all(np.isfinite(poly)):
                        continue
                    pp = to_plot(poly)
                    items.append(dpg.draw_polyline(pp, closed=True, color=colors[L], thickness=2, parent=plot))
                    tl, tr, bl = np.array(pp[0]), np.array(pp[n]), np.array(pp[3 * n])
                    f = 0.06
                    quad = [tl, tl + f * (tr - tl), tl + f * (tr - tl) + f * (bl - tl), tl + f * (bl - tl), tl]
                    items.append(dpg.draw_polygon(
                        [q.tolist() for q in quad], color=colors[L], fill=(*colors[L][:3], 150), parent=plot,
                    ))
                    p = sn["params"]
                    label = f"{L}  x{p['scale']:.3f}  rot {p['rot']:.2f}"
                    items.append(dpg.draw_text(
                        (tl + 0.02 * (tr - tl)).tolist(), label, color=colors[L], size=16, parent=plot,
                    ))

            if g["show_points"]:
                world_pts: dict[str, np.ndarray] = {}
                for L in ("A", "B"):
                    sn = snaps[L]
                    li = sn["image"]
                    if li is None or not sn["points"]:
                        continue
                    if mode == "Overlay":
                        world_pts[L] = eng.src_to_world(np.array(sn["points"]), sn["params"], sn["fit_H"], li.w, li.h)
                    elif mode[0] == L:
                        world_pts[L] = np.array(sn["points"], dtype=float)
                if mode == "Overlay" and "A" in world_pts and "B" in world_pts:
                    for pa, pb in zip(to_plot(world_pts["A"]), to_plot(world_pts["B"])):
                        items.append(dpg.draw_line(pa, pb, color=self.PAIR_COLOR, thickness=1, parent=plot))
                for L, pts in world_pts.items():
                    for i, (px, py) in enumerate(to_plot(pts)):
                        is_sel = (self._selected_point == (L, i))
                        pt_color = (255, 235, 50, 255) if is_sel else colors[L]
                        th = 3 if is_sel else 2
                        m = marker * 1.5 if is_sel else marker
                        if is_sel:
                            items.append(dpg.draw_circle([px, py], m * 1.4, color=(255, 235, 50, 255), thickness=2, parent=plot))
                        items.append(dpg.draw_line(
                            [px - m, py], [px + m, py], color=pt_color, thickness=th, parent=plot))
                        items.append(dpg.draw_line(
                            [px, py - m], [px, py + m], color=pt_color, thickness=th, parent=plot))
                        items.append(dpg.draw_text(
                            [px + m + 2, py + m + 2], f"{L}{i + 1}", color=pt_color, size=16 if is_sel else 14, parent=plot))
        except Exception as e:  # noqa: BLE001
            logger.debug(f"OverlayAligner overlay update failed: {e}")

    # ------------------------------------------------------------------
    # Emit / export
    # ------------------------------------------------------------------
    def _emit_visible_frame(
        self, bgr: np.ndarray, a: np.ndarray, g: dict[str, Any], save_path: str | None = None, emit: bool = True,
    ) -> None:
        """Emit the currently visible frame downstream (as transparent BGRA or with active background) and/or save to disk."""
        bg_mode = str(g["background"]) if g.get("export_background") else None
        custom_rgb = g.get("bg_custom")
        bgra = eng.to_bgra(bgr, a, background=bg_mode, custom_rgb=custom_rgb)

        if save_path:
            ok, buf = cv2.imencode(".png", bgra)
            if ok:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                buf.tofile(save_path)
                logger.success(f"OverlayAligner: saved {save_path}")

        if emit:
            for module in self.connections.get("RGBA", []):
                if hasattr(module, "input_cb"):
                    try:
                        module.input_cb(data=bgra, data_type=IOTypes.FRAME)
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"OverlayAligner emit error: {e}")

        self._set_text(
            "export_status",
            f"{'Emitted' if emit else 'Saved'} {bgra.shape[1]}x{bgra.shape[0]} ({g.get('view_mode', 'Visible')})"
            + (f" -> {os.path.basename(save_path)}" if save_path else ""),
        )

    def _schedule_emit(self, delay: float = 0.35) -> None:
        """Debounced auto-emit."""
        if self._emit_timer is not None:
            self._emit_timer.cancel()
        self._emit_timer = threading.Timer(delay, lambda: self._emit_worker(auto=True))
        self._emit_timer.daemon = True
        self._emit_timer.start()

    def _emit_async(self, force: bool = True) -> None:
        threading.Thread(target=self._emit_worker, kwargs={"force": force}, daemon=True).start()

    def _emit_worker(
        self, auto: bool = False, force: bool = True, emit: bool = True, save_path: str | None = None,
    ) -> None:
        """Emit the visible frame to outputs and/or save a PNG."""
        if not self._emit_lock.acquire(blocking=force):
            return
        try:
            with self._lock:
                bgr = self._current_rendered_bgr
                a = self._current_rendered_a
                g = dict(self.g)
            if bgr is None or a is None:
                self._set_text("export_status", "Nothing to emit (no frame rendered).")
                return

            self._emit_visible_frame(bgr, a, g, save_path=save_path, emit=emit)
        except Exception as e:  # noqa: BLE001
            logger.error(f"OverlayAligner export failed: {e}")
            self._set_text("export_status", f"Export failed: {e}")
        finally:
            self._emit_lock.release()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def is_ready(self) -> bool:
        """Always ready to receive frames."""
        return self.running

    def is_outputs_ready(self) -> bool:
        """Outputs are never blocking."""
        return True

    def on_close(self) -> None:
        """Stop worker threads and release GPU textures."""
        self.running = False
        if self._emit_timer is not None:
            self._emit_timer.cancel()
        self._dirty.set()
        if hasattr(self, "worker_thread") and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=0.3)
        self._clear_overlays()
        for tag in (self._t("handlers"), self._t("roi_rect"), self._t("roi_annot"), self._series, self._tex, self._tex_reg):
            try:
                if dpg.does_item_exist(tag):
                    dpg.delete_item(tag)
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        """Close the window and unregister."""
        self.on_close()
        super().close()


EXPORTED_CLASS = OverlayAlignerWin
EXPORTED_NAME = "Overlay Aligner"
