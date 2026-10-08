from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from typing import Any

import dearpygui.dearpygui as dpg
import numpy as np
from loguru import logger

from config.theme_manager import highlight
from core.app_state import app_state
from core.input_output_types import IOTypes
from core.window_base import WindowBase
from .serial_control import SerialControl
from .fluo_calibration_win import FluoCalibrationWin, CalibrationModelFit


class FluoControllerWin(WindowBase):
    """
    FluoControllerWin:
    Exposes control over the fluorescence hardware controller via serial.
    Allows adjusting Actinic, Detection, Far-Red, and White LED parameters.
    Supports running standardized test sequences and Fv/Fm measurements.
    """

    # Instance variables
    serial: SerialControl
    is_reading: bool
    read_thread: threading.Thread
    act_cal_loaded: bool
    act_threshold: int
    popt_low: tuple[float, float]
    popt_high: tuple[float, float]
    act_max_ue: float
    accepted_input_types: list[IOTypes]
    outputs: dict[str, IOTypes]
    connections: dict[str, list]
    _persistent_fields: list[str]
    combo_ports_tag: str
    btn_connect_tag: str
    console_tag: str
    act_tag: str
    det_tag: str
    fr_tag: str
    expostag: str
    intensetag: str
    act_ue_tag: str
    act_pwm_tag: str
    test_det_dur_tag: str
    test_det_dim_tag: str
    white_tag: str
    white2_tag: str
    fvfm_set_t_tag: str
    fvfm_dur_tag: str
    fvfm_int_tag: str
    pulse_delay_int_tag: str
    pulse_delay_exp_tag: str
    cal_label_tag: str
    sin_freq_tag: str
    sin_amp_tag: str
    calib_dialog: FluoCalibrationWin | None

    def __init__(
        self,
        label: str = "Fluo Controller",
        win_width: int = 560,
        win_height: int = 440,
        pos: tuple[int, int] = (0, 50),
        uuid: int | None = None,
        outputs: dict[str, IOTypes] | None = None,
        visible: bool = True,
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

        self.serial = SerialControl()
        self.serial.on_disconnect = self._on_serial_disconnected
        self.is_reading = True
        self.calib_dialog = None

        # Validation requirements
        self.accepted_input_types = [IOTypes.TEXT, IOTypes.CMD_DICT]
        self.outputs = {"received_data": IOTypes.TEXT, "control": IOTypes.CMD_DICT}
        self.connections = {k: [] for k in self.outputs}
        self._persistent_fields = ["label"]

        # Tags
        self.combo_ports_tag = f"fluo_ports_{self.UUID}"
        self.btn_connect_tag = f"fluo_connect_{self.UUID}"
        self.console_tag = f"fluo_console_{self.UUID}"
        self.act_tag = f"fluo_act_{self.UUID}"
        self.det_tag = f"fluo_det_{self.UUID}"
        self.fr_tag = f"fluo_fr_{self.UUID}"
        self.expostag = f"fluo_expos_{self.UUID}"
        self.intensetag = f"fluo_intense_{self.UUID}"
        self.act_ue_tag = f"fluo_act_ue_{self.UUID}"
        self.act_pwm_tag = f"fluo_act_pwm_{self.UUID}"
        self.test_det_dur_tag = f"test_det_dur_{self.UUID}"
        self.test_det_dim_tag = f"test_det_dim_{self.UUID}"
        self.white_tag = f"fluo_white_{self.UUID}"
        self.white2_tag = f"fluo_white2_{self.UUID}"

        self.fvfm_set_t_tag = f"fvfm_set_t_{self.UUID}"
        self.fvfm_dur_tag = f"fvfm_dur_{self.UUID}"
        self.fvfm_int_tag = f"fvfm_int_{self.UUID}"
        self.pulse_delay_int_tag = f"fluo_pulse_delay_int_{self.UUID}"
        self.pulse_delay_exp_tag = f"fluo_pulse_delay_exp_{self.UUID}"
        self.cal_label_tag = f"fluo_cal_label_{self.UUID}"
        self.sin_freq_tag = f"fluo_sin_freq_{self.UUID}"
        self.sin_amp_tag = f"fluo_sin_amp_{self.UUID}"

        # Actinic calibration (µE mode)
        self._load_act_calibration()

        if app_state.get("detector") is None:
            app_state.detector = 250

        # Build UI
        with dpg.window(
            label=self.label,
            width=self.win_width,
            height=self.win_height,
            pos=self.pos,
            tag=self.winID,
            show=self.visible,
            on_close=self.on_close,
        ):
            # Connection Controls
            with dpg.group(horizontal=True):
                dpg.add_button(label="Refresh", callback=self._refresh_ports)
                dpg.add_combo(items=[], width=100, tag=self.combo_ports_tag)
                dpg.add_button(label="Connect", tag=self.btn_connect_tag, callback=self._toggle_connect)
                dpg.add_text("Cal:")
                dpg.add_button(label="Browse", callback=self._on_browse_calibration)
                dpg.add_button(label="Calibrate", callback=self._open_calibration_tool)
                dpg.add_text("fluo_calibration.json", tag=self.cal_label_tag)

            dpg.add_separator()

            dpg.add_button(
                label="EMERGENCY STOP",
                width=-1,
                callback=lambda: self.serial.send_command("act 0 \n det 0 \n fr 0\n"),
            )

            test_det_grp = f"test_det_grp_{self.UUID}"
            with dpg.child_window(tag=test_det_grp, border=True, auto_resize_y=True, width=-1):
                dpg.add_text("Test Detection")
                with dpg.group(horizontal=True):
                    dpg.add_text("Dur:")
                    dpg.add_drag_int(
                        default_value=250, tag=self.test_det_dur_tag, width=70, callback=self._on_test_param_changed
                    )
                    dpg.add_text("Dim:")
                    dpg.add_drag_int(
                        default_value=512,
                        tag=self.test_det_dim_tag,
                        width=70,
                        min_value=0,
                        max_value=512,
                        clamped=True,
                        callback=self._on_test_param_changed,
                    )
                    dpg.add_button(label="Start", callback=self._run_test_detection)
            highlight(test_det_grp)

            with dpg.group(horizontal=True):
                dpg.add_text("Act")
                dpg.add_drag_int(
                    tag=self.act_tag,
                    default_value=0,
                    min_value=0,
                    clamped=True,
                    max_value=512,
                    width=80,
                    callback=lambda s, a: self.serial.send_command(
                        f"act {a} {dpg.get_value(self.act_pwm_tag) if dpg.does_item_exist(self.act_pwm_tag) else 512}\n"
                    ) if a is not None else None,
                )

                dpg.add_text("PWM")
                dpg.add_drag_int(
                    tag=self.act_pwm_tag,
                    default_value=512,
                    min_value=0,
                    clamped=True,
                    max_value=512,
                    width=80,
                    callback=lambda s, a: self.serial.send_command(
                        f"act {dpg.get_value(self.act_tag) if dpg.does_item_exist(self.act_tag) else 0} {a}\n"
                    ) if a is not None else None,
                )

                if self.act_cal_loaded:
                    with dpg.group(horizontal=True):
                        dpg.add_text("µE")
                        dpg.add_drag_int(
                            tag=self.act_ue_tag,
                            default_value=0,
                            min_value=0,
                            max_value=int(self.act_max_ue),
                            width=80,
                            clamped=True,
                            callback=self._on_act_ue_apply,
                        )

            with dpg.group(horizontal=True):
                dpg.add_text("Det")
                dpg.add_drag_int(
                    tag=self.det_tag,
                    default_value=0,
                    min_value=0,
                    clamped=True,
                    max_value=512,
                    width=80,
                    callback=self._on_det_changed,
                )
                dpg.add_text("Fr")
                dpg.add_drag_int(
                    tag=self.fr_tag,
                    default_value=0,
                    min_value=0,
                    width=80,
                    callback=lambda s, a: self.serial.send_command(f"fr {a}\n") if a is not None else None,
                )
                dpg.add_text("Wh")
                dpg.add_drag_int(
                    tag=self.white_tag,
                    default_value=0,
                    min_value=0,
                    max_value=512,
                    width=80,
                    callback=lambda s, a: self.serial.send_command(f"white {a}\n") if a is not None else None,
                )
                dpg.add_text("Wh2")
                dpg.add_drag_int(
                    tag=self.white2_tag,
                    default_value=0,
                    min_value=0,
                    max_value=512,
                    width=80,
                    callback=lambda s, a: self.serial.send_command(f"white2 {a}\n") if a is not None else None,
                )

            dpg.add_separator()

            dpg.add_text("Fv/Fm Test")
            with dpg.group(horizontal=True):
                dpg.add_text("Pulse:")
                dpg.add_drag_int(default_value=250, width=70, min_value=1, tag=self.fvfm_set_t_tag)
                dpg.add_text("Det:")
                dpg.add_drag_int(
                    default_value=250,
                    width=70,
                    min_value=1,
                    tag=self.fvfm_dur_tag,
                    callback=self._on_test_param_changed,
                )
                dpg.add_text("DInt:")
                dpg.add_drag_int(default_value=512, width=70, min_value=0, max_value=512, tag=self.fvfm_int_tag)
                dpg.add_button(label="Start", callback=self._run_fvfm_test)

            dpg.add_separator()

            dpg.add_text("Fixed int. variable exp. test")
            with dpg.group(horizontal=True):
                dpg.add_drag_int(
                    default_value=512, tag=self.intensetag, width=120, min_value=10, max_value=512, clamped=True
                )
                dpg.add_button(label="Start", callback=self._run_fixed_int_test)
                dpg.add_drag_int(default_value=250, tag=self.pulse_delay_int_tag, width=80, label="Gap (ms)")

            dpg.add_text("Fixed exp. variable int. test")
            with dpg.group(horizontal=True):
                dpg.add_drag_int(
                    min_value=10,
                    max_value=5000,
                    default_value=250,
                    width=120,
                    tag=self.expostag,
                    callback=self._on_test_param_changed,
                )
                dpg.add_button(label="Start", callback=self._run_fixed_exp_test)
                dpg.add_drag_int(default_value=250, tag=self.pulse_delay_exp_tag, width=80, label="Gap (ms)")

            dpg.add_separator()

            # Sin control
            dpg.add_text("Sin control")
            with dpg.group(horizontal=True):
                dpg.add_text("Sin Freq")
                dpg.add_drag_int(
                    tag=self.sin_freq_tag,
                    default_value=0,
                    min_value=0,
                    clamped=True,
                    max_value=512,
                    width=80,
                    callback=lambda s, a: self._send_sineact(),
                )
                dpg.add_text("Amp")
                dpg.add_drag_int(
                    tag=self.sin_amp_tag,
                    default_value=0,
                    min_value=0,
                    clamped=True,
                    max_value=512,
                    width=80,
                    callback=lambda s, a: self._send_sineact(),
                )
                dpg.add_button(label="Send", callback=self._send_sineact)
                dpg.add_button(label="Stop", callback=self._stop_sineact)

            dpg.add_separator()

            # Console log
            dpg.add_text("Console Log:")
            dpg.add_input_text(
                tag=self.console_tag,
                multiline=True,
                readonly=True,
                width=-1,
                height=100,
                default_value="",
            )

        self.autosize_window()
        self._refresh_ports()

        # Start reading thread
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()

        self.update_permission()

    def update_permission(self) -> None:
        """
        Adjust module permissions and UI visibility based on app_state.mode.
        Hide the window if mode is 'user'. Visible only in 'advanced' and 'dev'.
        """
        from core.app_state import app_state

        is_user = app_state.mode == "user"

        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def _refresh_ports(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        ports = self.serial.get_available_ports()
        dpg.configure_item(self.combo_ports_tag, items=ports)
        if ports:
            current = dpg.get_value(self.combo_ports_tag)
            if not current or current not in ports:
                dpg.set_value(self.combo_ports_tag, ports[0])

    def _set_button_color(self, tag: str, state: str) -> None:
        """
        Set button color theme based on state:
        'orange' (connecting / in progress),
        'green' (connected / success),
        'red' (disconnected / failed / error),
        'default' (reset theme).
        """
        if not dpg.does_item_exist(tag):
            return

        if state == "orange":
            with dpg.theme() as theme:
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (255, 165, 0), category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 180, 50), category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 140, 0), category=dpg.mvThemeCat_Core)
            dpg.bind_item_theme(tag, theme)
        elif state == "green":
            with dpg.theme() as theme:
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 100, 0), category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 150, 0), category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 200, 0), category=dpg.mvThemeCat_Core)
            dpg.bind_item_theme(tag, theme)
        elif state == "red":
            with dpg.theme() as theme:
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (150, 0, 0), category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (200, 0, 0), category=dpg.mvThemeCat_Core)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 0, 0), category=dpg.mvThemeCat_Core)
            dpg.bind_item_theme(tag, theme)
        elif state == "default":
            dpg.bind_item_theme(tag, 0)

    def _on_serial_disconnected(self) -> None:
        if dpg.does_item_exist(self.btn_connect_tag):
            dpg.configure_item(self.btn_connect_tag, label="Connect")
        self._set_button_color(self.btn_connect_tag, "red")
        self._log_to_console("Serial disconnected.")

    def _toggle_connect(self, *args: Any, **kwargs: Any) -> None:
        btn_label = dpg.get_item_label(self.btn_connect_tag) if dpg.does_item_exist(self.btn_connect_tag) else ""
        if self.serial.connected or btn_label == "Disconnect":
            self._disconnect()
        else:
            self._connect()

    def _connect(self, port: str | None = None) -> None:
        if port:
            if dpg.does_item_exist(self.combo_ports_tag):
                dpg.set_value(self.combo_ports_tag, port)
        else:
            port = dpg.get_value(self.combo_ports_tag) if dpg.does_item_exist(self.combo_ports_tag) else None

        if not port:
            self._set_button_color(self.btn_connect_tag, "red")
            self._log_to_console("Error: No port selected.")
            return

        self._set_button_color(self.btn_connect_tag, "orange")
        self._log_to_console(f"Connecting to {port}...")
        if self.serial.connect(port):
            if dpg.does_item_exist(self.btn_connect_tag):
                dpg.configure_item(self.btn_connect_tag, label="Disconnect")
            self._set_button_color(self.btn_connect_tag, "green")
            self._log_to_console(f"Connected to {port}.")
        else:
            if dpg.does_item_exist(self.btn_connect_tag):
                dpg.configure_item(self.btn_connect_tag, label="Connect")
            self._set_button_color(self.btn_connect_tag, "red")
            self._log_to_console(f"Connection to {port} failed.")

    def _disconnect(self) -> None:
        if dpg.does_item_exist(self.btn_connect_tag):
            dpg.configure_item(self.btn_connect_tag, label="Connect")
        self._set_button_color(self.btn_connect_tag, "red")
        self._log_to_console("Disconnecting...")
        self.serial.disconnect()

    def _load_act_calibration(self, cal_path: str | None = None) -> bool:
        """Load actinic calibration data from JSON and fit curves."""
        self.act_cal_loaded = False
        self.act_threshold = 0
        self.act_max_ue = 0.0
        self.low_inv_fn = None
        self.high_inv_fn = None

        if cal_path is None:
            cal_path = os.path.join(os.path.dirname(__file__), "fluo_calibration.json")
        else:
            if not os.path.exists(cal_path):
                # Try relative to the module's directory
                candidate = os.path.join(os.path.dirname(__file__), cal_path)
                if os.path.exists(candidate):
                    cal_path = candidate

        if not os.path.exists(cal_path):
            logger.warning(f"Fluo calibration file not found: {cal_path}")
            self._log_to_console(f"Error: Calibration file not found: {cal_path}")
            return False

        try:
            with open(cal_path, "r", encoding="utf-8") as f:
                cal_data = json.load(f)

            act_cal = cal_data.get("act_calibration")
            if not act_cal:
                logger.warning(f"No 'act_calibration' key in {cal_path}")
                self._log_to_console(f"Error: No 'act_calibration' key in {os.path.basename(cal_path)}")
                return False

            self.act_threshold = act_cal.get("threshold", 150)

            # Low regime model fit
            low = act_cal.get("low_cal", {})
            pwm_low = [float(p) for p in low.get("pwm", [])]
            int_low = [float(i) for i in low.get("intensity", [])]

            low_m_obj = low.get("model")
            if isinstance(low_m_obj, dict):
                low_m_type = low_m_obj.get("type", "power")
                low_deg = int(low_m_obj.get("degree", 2))
            elif isinstance(low_m_obj, str):
                low_m_type = low_m_obj
                low_deg = 2
            else:
                low_m_type = "power"
                low_deg = 2

            low_ready, eq_low, r2_low, fwd_low, inv_low, _ = CalibrationModelFit.fit(
                pwm_low, int_low, low_m_type, low_deg
            )
            if low_ready and inv_low is not None:
                self.low_inv_fn = inv_low
                logger.debug(f"Act Low fit ({low_m_type}): {eq_low} (R²={r2_low:.4f})")
            else:
                logger.warning(f"Failed to fit Low calibration: {eq_low}")

            # High regime model fit
            high = act_cal.get("high_cal", {})
            adim_high = [float(a) for a in high.get("adim", [])]
            int_high = [float(i) for i in high.get("intensity", [])]

            high_m_obj = high.get("model")
            if isinstance(high_m_obj, dict):
                high_m_type = high_m_obj.get("type", "linear")
                high_deg = int(high_m_obj.get("degree", 2))
            elif isinstance(high_m_obj, str):
                high_m_type = high_m_obj
                high_deg = 2
            else:
                high_m_type = "linear"
                high_deg = 2

            high_ready, eq_high, r2_high, fwd_high, inv_high, _ = CalibrationModelFit.fit(
                adim_high, int_high, high_m_type, high_deg
            )
            if high_ready and inv_high is not None:
                self.high_inv_fn = inv_high
                self.act_max_ue = float(max(int_high)) if int_high else 0.0
                logger.debug(f"Act High fit ({high_m_type}): {eq_high} (R²={r2_high:.4f})")
            else:
                logger.warning(f"Failed to fit High calibration: {eq_high}")

            self.act_cal_loaded = (self.low_inv_fn is not None and self.high_inv_fn is not None)
            if self.act_cal_loaded:
                logger.success(f"Actinic µE calibration loaded successfully from {cal_path}")

            # Dynamically update the slider limit in UI
            if hasattr(self, "act_ue_tag") and dpg.does_item_exist(self.act_ue_tag):
                dpg.configure_item(self.act_ue_tag, max_value=int(self.act_max_ue))

            filename = os.path.basename(cal_path)
            if hasattr(self, "cal_label_tag") and dpg.does_item_exist(self.cal_label_tag):
                dpg.set_value(self.cal_label_tag, filename)
            self._log_to_console(f"Calibration loaded: {filename}")
            return True

        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load fluo calibration: {e}")
            self._log_to_console(f"Error loading calibration: {e}")
            return False

    def _on_browse_calibration(self, *args: Any, **kwargs: Any) -> None:
        from core.file_explorer import file_explorer
        default_dir = os.path.dirname(__file__)
        file_explorer.select_file(
            default_path=default_dir,
            extensions=[("JSON files", "*.json"), ("All files", "*.*")],
            callback=self._on_calibration_file_selected
        )

    def _open_calibration_tool(self, *args: Any, **kwargs: Any) -> None:
        """Opens or focuses the Fluo Calibration Assistant window."""
        if self.calib_dialog is None:
            self.calib_dialog = FluoCalibrationWin(
                serial=self.serial,
                on_calibration_saved=self._load_act_calibration,
                parent_uuid=self.UUID,
            )
        else:
            self.calib_dialog.show()

    def _on_calibration_file_selected(self, file_path: str, ext: str, *args, **kwargs) -> None:
        if not file_path:
            return
        
        logger.info(f"Loading custom calibration file: {file_path}")
        self._load_act_calibration(file_path)

    def _set_act_uE(self, intensity: float | int) -> None:
        """Convert a µE intensity to ADIM/PWM and send to the actinic LED."""
        if not self.act_cal_loaded:
            self._log_to_console("Calibration not loaded, cannot set µE")
            return

        intensity_f = float(intensity)
        if intensity_f <= self.act_threshold:
            pwm_val = self.low_inv_fn(intensity_f) if self.low_inv_fn else 0.0
            adim = 512
            pwm = int(round(float(pwm_val)))
        else:
            adim_val = self.high_inv_fn(intensity_f) if self.high_inv_fn else 0.0
            adim = int(round(float(adim_val)))
            pwm = 512

        adim = max(0, min(512, adim))
        pwm = max(0, min(512, pwm))

        self.serial.send_command(f"act {adim} {pwm}\n")
        self._log_to_console(f"Act µE={intensity_f:.1f} → ADIM={adim}, PWM={pwm}")

    def _on_act_ue_apply(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Callback for the Act µE Apply button."""
        val = app_data if app_data is not None else (dpg.get_value(self.act_ue_tag) if dpg.does_item_exist(self.act_ue_tag) else None)
        if val is not None:
            try:
                self._set_act_uE(float(val))
            except (ValueError, TypeError):
                pass

    def _on_det_changed(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Callback for the Det slider change."""
        val = app_data if app_data is not None else (dpg.get_value(self.det_tag) if dpg.does_item_exist(self.det_tag) else None)
        if val is not None:
            try:
                app_state.detector = int(val)
            except (ValueError, TypeError):
                pass
            if self.serial.connected:
                self.serial.send_command(f"det {val}\n")

    def _read_loop(self) -> None:
        while self.is_reading:
            try:
                if self.serial.connected:
                    data = self.serial.read()
                    if data:
                        self._log_to_console(f"<< {data.strip()}")
            except Exception as e:
                logger.error(f"Error in FluoController read loop: {e}")
            time.sleep(0.1)

    def _run_test_detection(self, *args: Any, **kwargs: Any) -> None:
        """Runs 'testdet' command after capturing IDS reference frames."""
        raw_dur = dpg.get_value(self.test_det_dur_tag) if dpg.does_item_exist(self.test_det_dur_tag) else None
        raw_dim = dpg.get_value(self.test_det_dim_tag) if dpg.does_item_exist(self.test_det_dim_tag) else None

        dur = int(raw_dur) if raw_dur is not None else 250
        dim = int(raw_dim) if raw_dim is not None else 512

        cmd_str = f"detection {dur} {dim}\n"

        def run_seq() -> None:
            self._log_to_console("Capturing IDS reference frame...")
            self._emit_control(
                {
                    "trigger_mode": "Trigger",
                    "exposure": dur,
                    "start_dark_ref": True,
                    "n_frames": 10,
                }
            )
            time.sleep(10 * (dur / 1000000.0) + 2.0)

            if self.serial.connected:
                self.serial.send_command(cmd_str)
            self._log_to_console(f"Sent Test Detection cmd: {cmd_str.strip()}")

        threading.Thread(target=run_seq, daemon=True).start()

    def _run_fixed_exp_test(self, *args: Any, **kwargs: Any) -> None:
        """Standardized sequence for Fixed Exp, Variable Int test."""
        exp_range = range(50, 550, 50)
        raw_val = dpg.get_value(self.expostag) if dpg.does_item_exist(self.expostag) else None
        val = int(raw_val) if raw_val is not None else 250
        
        raw_gap = dpg.get_value(self.pulse_delay_exp_tag) if dpg.does_item_exist(self.pulse_delay_exp_tag) else None
        gap = (int(raw_gap) if raw_gap is not None else 250) / 1000.0
        group = "TInt"
        names = [f"{i}TInt_{i}" for i in exp_range]

        self._emit_control(
            {
                "trigger_mode": "Trigger",
                "exposure": val,
                "clear": True,
                "record": True,
                "save": False,
                "add_timestamp": False,
                "reset_ts": True,
                "group": group,
            }
        )
        self._log_to_console(f"Starting standardized Fixed Exp Test. Group: {group}")

        def run_seq() -> None:
            # 2. Dark Reference Capture (IDS + MCU)
            self._emit_control({"start_dark_ref": True, "n_frames": 10})

            # Wait for 10 frames + processing margin
            time.sleep(10 * (val / 1000000.0) + 2.0)

            # 3. Setup names for test frames
            self._emit_control({"names": names})

            # 4. Series of detections duration dim
            for intensity in exp_range:
                if not self.serial.connected or not self.is_reading:
                    break
                self.serial.send_command(f"detection {val} {intensity}\n")
                self._log_to_console(f"Flash: {val}µs Int:{intensity}")
                time.sleep(val / 1000000.0 + gap)

            self._log_to_console("Standardized test finished.")

        threading.Thread(target=run_seq, daemon=True).start()

    def _run_fixed_int_test(self, *args: Any, **kwargs: Any) -> None:
        """Standardized sequence for Fixed Int, Variable Exp test."""
        exp_range = range(50, 2550, 50)
        raw_int = dpg.get_value(self.intensetag) if dpg.does_item_exist(self.intensetag) else None
        intensity = int(raw_int) if raw_int is not None else 512
        
        raw_gap = dpg.get_value(self.pulse_delay_int_tag) if dpg.does_item_exist(self.pulse_delay_int_tag) else None
        gap = (int(raw_gap) if raw_gap is not None else 250) / 1000.0
        ref_exposure = 2500
        group = "TExp"
        names = [f"{i}_TExp_{i}" for i in exp_range]

        self._emit_control(
            {
                "trigger_mode": "Trigger",
                "exposure": ref_exposure,
                "clear": True,
                "record": True,
                "save": False,
                "add_timestamp": False,
                "reset_ts": True,
                "group": group,
            }
        )
        self._log_to_console(f"Starting standardized Fixed Int Test. Group: {group}")

        def run_seq() -> None:
            # 2. Dark Reference Capture
            self._emit_control({"start_dark_ref": True, "n_frames": 10})
            time.sleep(10 * (ref_exposure / 1000000.0) + 2.0)

            # 3. Setup names
            self._emit_control({"names": names})

            # 4. Series of detections duration dim
            for exposure in exp_range:
                if not self.serial.connected or not self.is_reading:
                    break
                self.serial.send_command(f"detection {exposure} {intensity}\n")
                self._log_to_console(f"Flash: {exposure}µs Int:{intensity}")
                time.sleep(exposure / 1000000.0 + gap)

            self._log_to_console("Standardized test finished.")

        threading.Thread(target=run_seq, daemon=True).start()

    def _run_fvfm_test(self, *args: Any, **kwargs: Any) -> None:
        """Runs 'fvfm' test."""
        raw_set_t = dpg.get_value(self.fvfm_set_t_tag) if dpg.does_item_exist(self.fvfm_set_t_tag) else None
        raw_dur = dpg.get_value(self.fvfm_dur_tag) if dpg.does_item_exist(self.fvfm_dur_tag) else None
        raw_int = dpg.get_value(self.fvfm_int_tag) if dpg.does_item_exist(self.fvfm_int_tag) else None

        set_t = int(raw_set_t) if raw_set_t is not None else 250
        dur = int(raw_dur) if raw_dur is not None else 250
        intens = int(raw_int) if raw_int is not None else 512

        cmd_str = f"fluo {set_t} {dur} 512 {intens}\n"
        self._emit_control({"exposure": dur})
        if self.serial.connected:
            self.serial.send_command(cmd_str)
        self._log_to_console(f"Sent Fv/Fm cmd: {cmd_str.strip()}")

    def _on_test_param_changed(self, sender: str, app_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Emits an exposure command when test parameters are modified."""
        if sender == self.test_det_dur_tag or sender == self.expostag:
            dur = app_data if app_data is not None else dpg.get_value(sender)
        elif sender == self.test_det_dim_tag:
            dur = dpg.get_value(self.test_det_dur_tag)
        elif sender == self.fvfm_dur_tag:
            dur = app_data if app_data is not None else dpg.get_value(self.fvfm_dur_tag)
        else:
            return

        if dur is None:
            return

        try:
            dur_val = int(dur)
        except (ValueError, TypeError):
            return

        # Synchronize all tags
        for tag in [self.test_det_dur_tag, self.fvfm_dur_tag, self.expostag]:
            if tag != sender and dpg.does_item_exist(tag):
                dpg.set_value(tag, dur_val)

        app_state.detector = dur_val
        cmd = {"exposure": dur_val}
        self._emit_control(cmd)
        self._log_to_console(f"Auto-emit exposure: {dur_val} µs")

    def _handle_ref(self, n_frames: int | None = None, exposure: int | None = None) -> None:
        """Setup the camera then trigger its autonomous REF capture, and trigger the MCU."""
        if n_frames is None:
            n_frames = 10

        if exposure is not None:
            dpg.set_value(self.expostag, int(exposure))
            self._on_test_param_changed(self.expostag, int(exposure))

        def task() -> None:
            active_exposure = dpg.get_value(self.expostag) if dpg.does_item_exist(self.expostag) else 2000

            # 1. Setup Camera for Dark Ref
            self._emit_control({"trigger_mode": "Trigger", "exposure": int(active_exposure)})

            # 2. Wait for camera to apply settings
            time.sleep(0.15)

            # 3. Tell IDS to start its autonomous dark ref capture
            self._emit_control({"start_dark_ref": True, "n_frames": int(n_frames)})

        threading.Thread(target=task, daemon=True).start()

    def _emit_control(self, cmd_dict: dict[str, Any]) -> None:
        for module in self.connections.get("control", []):
            if hasattr(module, "input_cb"):
                module.input_cb(data=cmd_dict, data_type=IOTypes.CMD_DICT)

    def _log_to_console(self, text: str) -> None:
        if hasattr(self, "console_tag") and dpg.does_item_exist(self.console_tag):
            current = dpg.get_value(self.console_tag)
            timestamp = datetime.now().strftime("[%H:%M:%S] ")
            new_text = f"{current}\n{timestamp}{text}" if current else f"{timestamp}{text}"
            if len(new_text) > 5000:
                new_text = new_text[-5000:]
            dpg.set_value(self.console_tag, new_text)

    def input_cb(self, *args: Any, **kwargs: Any) -> None:
        """Callback for receiving data from other nodes."""
        data = kwargs.get("data") if "data" in kwargs else (args[0] if args else None)
        data_type = kwargs.get("data_type")

        if data_type == IOTypes.CMD_DICT or isinstance(data, dict):
            # Command dictionaries from sequence processor
            cmd_payload = kwargs.get("cmd") if kwargs.get("cmd") is not None else data
            if isinstance(cmd_payload, dict) and "send" in cmd_payload:
                if self.serial.connected:
                    cmd_str = cmd_payload["send"]
                    if not cmd_str.endswith("\n"):
                        cmd_str += "\n"
                    self.serial.send_command(cmd_str)
                    self._log_to_console(f"Auto >> {cmd_str.strip()}")

            elif isinstance(cmd_payload, dict) and "connect" in cmd_payload:
                val = cmd_payload["connect"]
                if val is False:
                    self._disconnect()
                else:
                    port = val if isinstance(val, str) else cmd_payload.get("port")
                    self._connect(port=port)

            elif isinstance(cmd_payload, dict) and "disconnect" in cmd_payload:
                self._disconnect()

            elif isinstance(cmd_payload, dict) and (
                "load_calibration" in cmd_payload
                or "load_cal" in cmd_payload
                or "calibration" in cmd_payload
            ):
                cal_target = (
                    cmd_payload.get("load_calibration")
                    or cmd_payload.get("load_cal")
                    or cmd_payload.get("calibration")
                )
                if isinstance(cal_target, str):
                    cal_path = cal_target
                else:
                    cal_path = cmd_payload.get("path") or cmd_payload.get("file")
                self._load_act_calibration(cal_path)

            elif isinstance(cmd_payload, dict) and "fluo_ref" in cmd_payload:
                n_frames = cmd_payload.get("n_frames")
                exposure = cmd_payload.get("exposure")
                self._handle_ref(n_frames=n_frames, exposure=exposure)

            elif isinstance(cmd_payload, dict) and "trigger_cam" in cmd_payload:
                # Proxy trigger to camera
                self._emit_control(cmd_payload)

            elif isinstance(cmd_payload, dict) and "act" in cmd_payload:
                intensity = cmd_payload["act"]
                self._set_act_uE(intensity)
                self._log_to_console(f"Auto >> Act µE={intensity}")

            elif isinstance(cmd_payload, dict) and "fvfm" in cmd_payload:
                self._run_fvfm_test()
                self._log_to_console("Auto >> Fv/Fm measurement triggered")
            elif isinstance(cmd_payload, dict) and "white" in cmd_payload:
                intensity = int(cmd_payload["white"])
                if dpg.does_item_exist(self.white_tag):
                    dpg.set_value(self.white_tag, intensity)
                if self.serial.connected:
                    self.serial.send_command(f"white {intensity}\n")
                self._log_to_console(f"Auto >> White={intensity}")

            elif isinstance(cmd_payload, dict) and "white2" in cmd_payload:
                intensity = int(cmd_payload["white2"])
                if dpg.does_item_exist(self.white2_tag):
                    dpg.set_value(self.white2_tag, intensity)
                if self.serial.connected:
                    self.serial.send_command(f"white2 {intensity}\n")
                self._log_to_console(f"Auto >> White2={intensity}")

            elif isinstance(cmd_payload, dict) and "testdet" in cmd_payload:
                self._run_test_detection()
                self._log_to_console("Auto >> Test detection triggered")
            elif isinstance(cmd_payload, dict) and "fluo_fixed_int_test" in cmd_payload:
                self._run_fixed_int_test()
                self._log_to_console("Auto >> Fixed Int Var Exp test triggered")
            elif isinstance(cmd_payload, dict) and "exposure" in cmd_payload:
                val = int(cmd_payload["exposure"])
                app_state.detector = val
                if dpg.does_item_exist(self.test_det_dur_tag):
                    dpg.set_value(self.test_det_dur_tag, val)
                if dpg.does_item_exist(self.fvfm_dur_tag):
                    dpg.set_value(self.fvfm_dur_tag, val)
                if dpg.does_item_exist(self.expostag):
                    dpg.set_value(self.expostag, val)
                self._log_to_console(f"Auto >> Expositions updated to {val} µs")

            elif isinstance(cmd_payload, dict) and ("det" in cmd_payload or "detector" in cmd_payload):
                raw_det = cmd_payload.get("det", cmd_payload.get("detector"))
                try:
                    val = int(raw_det)
                    app_state.detector = val
                    if dpg.does_item_exist(self.det_tag):
                        dpg.set_value(self.det_tag, val)
                    if self.serial.connected:
                        self.serial.send_command(f"det {val}\n")
                    self._log_to_console(f"Auto >> Det updated to {val}")
                except (ValueError, TypeError):
                    pass

            elif isinstance(cmd_payload, dict) and "sineact" in cmd_payload:
                sine_val = cmd_payload["sineact"]
                if isinstance(sine_val, (list, tuple)) and len(sine_val) >= 2:
                    freq = float(sine_val[0])
                    amp = int(sine_val[1])
                elif isinstance(sine_val, dict):
                    freq = float(sine_val.get("freq", 1.0))
                    amp = int(sine_val.get("amp", 256))
                else:
                    freq = float(sine_val)
                    amp = int(cmd_payload.get("amp", 256))

                if dpg.does_item_exist(self.sin_freq_tag):
                    dpg.set_value(self.sin_freq_tag, int(freq))
                if dpg.does_item_exist(self.sin_amp_tag):
                    dpg.set_value(self.sin_amp_tag, int(amp))

                cmd_str = f"sineact {int(freq)} {int(amp)}\n"
                if self.serial.connected:
                    self.serial.send_command(cmd_str)
                self._log_to_console(f"Auto >> sineact {int(freq)} {int(amp)}")

    def _send_sineact(self, *args: Any, **kwargs: Any) -> None:
        """Sends 'sineact [FREQ] [Amplitude]' command."""
        freq = dpg.get_value(self.sin_freq_tag)
        amp = dpg.get_value(self.sin_amp_tag)

        cmd_str = f"sineact {freq} {amp}\n"
        if self.serial.connected:
            self.serial.send_command(cmd_str)
        self._log_to_console(f"Sent Sine Act cmd: {cmd_str.strip()}")

    def _stop_sineact(self, *args: Any, **kwargs: Any) -> None:
        """Sends 'sineact 0 0' to stop sinusoidal mode."""
        cmd_str = "sineact 0 0\n"
        if self.serial.connected:
            self.serial.send_command(cmd_str)
        self._log_to_console("Stopped Sine Act (sineact 0 0)")

    def on_close(self) -> None:
        self.is_reading = False
        self._disconnect()
        if self.calib_dialog is not None:
            self.calib_dialog.on_close()
        if dpg.does_item_exist(self.winID):
            dpg.delete_item(self.winID)

    def close(self) -> None:
        self.is_reading = False
        self._disconnect()
        if self.calib_dialog is not None:
            self.calib_dialog.on_close()
        super().close()


EXPORTED_CLASS = FluoControllerWin
EXPORTED_NAME = "Fluo_controller"
