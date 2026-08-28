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

from core.app_state import app_state
from core.input_output_types import IOTypes
from core.window_base import WindowBase
from .serial_control import SerialControl


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
        self.is_reading = True

        # Actinic calibration (µE mode)
        self._load_act_calibration()

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
                dpg.add_text("  Cal:")
                dpg.add_button(label="Browse", callback=self._on_browse_calibration)
                dpg.add_text("fluo_calibration.json", tag=self.cal_label_tag)

            dpg.add_separator()

            dpg.add_button(
                label="EMERGENCY STOP",
                width=-1,
                callback=lambda: self.serial.send_command("act 0 \n det 0 \n fr 0\n"),
            )

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
                    callback=lambda s, a: self.serial.send_command(f"det {a}\n") if a is not None else None,
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

    def _toggle_connect(self, *args: Any, **kwargs: Any) -> None:
        if self.serial.connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self) -> None:
        port = dpg.get_value(self.combo_ports_tag)
        if not port:
            self._log_to_console("Error: No port selected.")
            return

        self._log_to_console(f"Connecting to {port}...")
        if self.serial.connect(port):
            dpg.configure_item(self.btn_connect_tag, label="Disconnect")
            self._log_to_console("Connected.")
        else:
            self._log_to_console("Connection failed.")

    def _disconnect(self) -> None:
        self.serial.disconnect()
        dpg.configure_item(self.btn_connect_tag, label="Connect")
        self._log_to_console("Disconnected.")

    def _load_act_calibration(self, cal_path: str | None = None) -> None:
        """Load actinic calibration data from JSON and fit curves."""
        self.act_cal_loaded = False
        self.act_threshold = 0
        self.popt_low = (1.0, 1.0)
        self.popt_high = (1.0, 0.0)
        self.act_max_ue = 0.0

        if cal_path is None:
            cal_path = os.path.join(os.path.dirname(__file__), "fluo_calibration.json")

        if not os.path.exists(cal_path):
            logger.warning(f"Fluo calibration file not found: {cal_path}")
            return

        try:
            with open(cal_path, "r", encoding="utf-8") as f:
                cal_data = json.load(f)

            act_cal = cal_data.get("act_calibration")
            if not act_cal:
                logger.warning(f"No 'act_calibration' key in {cal_path}")
                return

            self.act_threshold = act_cal.get("threshold", 150)

            # Low regime: power-law fit intensity = a * pwm^b
            low = act_cal.get("low_cal", {})
            pwm_low = np.array(low.get("pwm", []), dtype=float)
            int_low = np.array(low.get("intensity", []), dtype=float)

            if len(pwm_low) >= 2 and len(int_low) >= 2:
                mask = (pwm_low > 0) & (int_low > 0)
                log_pwm = np.log(pwm_low[mask])
                log_int = np.log(int_low[mask])
                b, log_a = np.polyfit(log_pwm, log_int, 1)
                a = np.exp(log_a)
                self.popt_low = (float(a), float(b))
                logger.debug(f"Act low-regime fit: a={a:.4f}, b={b:.4f}")
            else:
                logger.warning("Not enough low_cal data points for fitting")

            # High regime: linear fit intensity = a * adim + b
            high = act_cal.get("high_cal", {})
            adim_high = np.array(high.get("adim", []), dtype=float)
            int_high = np.array(high.get("intensity", []), dtype=float)

            if len(adim_high) >= 2 and len(int_high) >= 2:
                a, b = np.polyfit(adim_high, int_high, 1)
                self.popt_high = (float(a), float(b))
                logger.debug(f"Act high-regime fit: a={a:.4f}, b={b:.4f}")
                self.act_max_ue = float(max(int_high))
            else:
                logger.warning("Not enough high_cal data points for fitting")

            self.act_cal_loaded = True
            logger.success(f"Actinic µE calibration loaded successfully from {cal_path}")

            # Dynamically update the slider limit in UI
            if hasattr(self, "act_ue_tag") and dpg.does_item_exist(self.act_ue_tag):
                dpg.configure_item(self.act_ue_tag, max_value=int(self.act_max_ue))

        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load fluo calibration: {e}")

    def _on_browse_calibration(self, *args: Any, **kwargs: Any) -> None:
        from core.file_explorer import file_explorer
        default_dir = os.path.dirname(__file__)
        file_explorer.select_file(
            default_path=default_dir,
            extensions=[("JSON files", "*.json"), ("All files", "*.*")],
            callback=self._on_calibration_file_selected
        )

    def _on_calibration_file_selected(self, file_path: str, ext: str) -> None:
        if not file_path:
            return
        
        logger.info(f"Loading custom calibration file: {file_path}")
        self._load_act_calibration(file_path)
        
        if self.act_cal_loaded:
            filename = os.path.basename(file_path)
            if dpg.does_item_exist(self.cal_label_tag):
                dpg.set_value(self.cal_label_tag, filename)
            self._log_to_console(f"Calibration loaded: {filename}")
        else:
            self._log_to_console("Error: Failed to load selected calibration file.")

    def _set_act_uE(self, intensity: float | int) -> None:
        """Convert a µE intensity to ADIM/PWM and send to the actinic LED."""
        if not self.act_cal_loaded:
            self._log_to_console("Calibration not loaded, cannot set µE")
            return

        if intensity <= self.act_threshold:
            a, b = self.popt_low
            pwm_val = np.power(intensity / a, 1.0 / b)
            adim = 512
            pwm = int(round(float(pwm_val)))
        else:
            a, b = self.popt_high
            adim_val = (intensity - b) / a
            adim = int(round(float(adim_val)))
            pwm = 512

        adim = max(0, min(512, adim))
        pwm = max(0, min(512, pwm))

        self.serial.send_command(f"act {adim} {pwm}\n")
        self._log_to_console(f"Act µE={intensity:.1f} → ADIM={adim}, PWM={pwm}")

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

    def _read_loop(self) -> None:
        while self.is_reading:
            if self.serial.connected:
                data = self.serial.read()
                if data:
                    self._log_to_console(f"<< {data.strip()}")
            time.sleep(0.1)

    def _run_test_detection(self, *args: Any, **kwargs: Any) -> None:
        """Runs 'testdet' command."""
        raw_dur = dpg.get_value(self.test_det_dur_tag) if dpg.does_item_exist(self.test_det_dur_tag) else None
        raw_dim = dpg.get_value(self.test_det_dim_tag) if dpg.does_item_exist(self.test_det_dim_tag) else None

        dur = int(raw_dur) if raw_dur is not None else 250
        dim = int(raw_dim) if raw_dim is not None else 512

        cmd_str = f"detection {dur} {dim}\n"
        self._emit_control({"exposure": dur})
        if self.serial.connected:
            self.serial.send_command(cmd_str)
        self._log_to_console(f"Sent Test Detection cmd: {cmd_str.strip()}")

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
        if dpg.does_item_exist(self.console_tag):
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
                self._connect()

            elif isinstance(cmd_payload, dict) and "disconnect" in cmd_payload:
                self._disconnect()

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
                if dpg.does_item_exist(self.test_det_dur_tag):
                    dpg.set_value(self.test_det_dur_tag, val)
                if dpg.does_item_exist(self.fvfm_dur_tag):
                    dpg.set_value(self.fvfm_dur_tag, val)
                if dpg.does_item_exist(self.expostag):
                    dpg.set_value(self.expostag, val)
                self._log_to_console(f"Auto >> Expositions updated to {val} µs")

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
        self.serial.disconnect()
        dpg.delete_item(self.winID)


EXPORTED_CLASS = FluoControllerWin
EXPORTED_NAME = "Fluo_controller"
