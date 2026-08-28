from __future__ import annotations
import dearpygui.dearpygui as dpg
from loguru import logger
from core.window_base import WindowBase
from core.input_output_types import IOTypes
import numpy as np
import os
import datetime
import json
from typing import Any

class TaraWin(WindowBase):
    """
    TARA Module: Simplified automation interface.
    """
    od50_config: dict[str, dict[str, Any]]
    naming_fields_keys: list[str]
    field_config: dict[str, dict[str, Any]]
    _persistent_fields: list[str]
    naming_values: dict[str, Any]
    target_signal: float
    protocol_step: int
    pipeline_mode: str | None
    current_folder: str | None
    master_folder: str | None
    working_folder: str | None
    last_date_folder: str | None
    _protocol_set_clicked: bool
    _metadata_set_clicked: bool
    _subfolders_created: bool
    _test_detection_clicked: bool
    _picture_taken: bool
    accepted_input_types: list[IOTypes]
    output_types: list[IOTypes]
    _find_exp_active: bool
    _find_exp_frame_count: int
    _find_exp_expected_frames: int
    _progress_popup_tag: str
    _progress_bar_tag: str
    _progress_text_tag: str
    outputs: dict[str, Any]
    connections: dict[str, list[Any]]
    
    # UI Tags
    start_protocol_btn_tag: str
    step1_group_tag: str
    step2_group_tag: str
    step3_group_tag: str
    step4_group_tag: str
    step5_group_tag: str
    step6_group_tag: str
    step8_group_tag: str
    step9_group_tag: str
    unified_next_btn_tag: str
    unified_prev_btn_tag: str
    skip_vba_btn_tag: str
    select_folder_btn_tag: str
    target_signal_tag: str
    find_best_exp_btn_tag: str
    best_exp_int_tag: str
    set_expos_btn_tag: str
    protocol_combo_tag: str
    run_protocol_btn_tag: str
    group_name_input_tag: str
    naming_toggle_tag: str
    complex_naming_group_tag: str
    simple_naming_group_tag: str
    naming_fields_tags: dict[str, str]
    naming_fields_custom_tags: dict[str, str]
    naming_fields_combo_tags: dict[str, str]
    protocol_tooltip_text_tag: str
    save_metadata_btn_tag: str
    start_sequence_btn_tag: str
    main_subfolder_tag: str
    fvfm_subfolder_tag: str
    superseq_subfolder_tag: str
    create_subfolders_btn_tag: str
    master_folder_btn_tag: str
    master_folder_label_tag: str
    working_folder_btn_tag: str
    working_folder_label_tag: str
    mode_select_group_tag: str
    capture_mode_btn_tag: str
    processing_mode_btn_tag: str
    processing_vba_mode_btn_tag: str
    proc_step1_group_tag: str
    proc_working_folder_btn_tag: str
    proc_working_folder_label_tag: str
    capture_done_group_tag: str
    switch_to_processing_btn_tag: str
    test_det_btn_tag: str
    capture_done_prev_btn_tag: str
    new_exp_btn_tag: str
    close_software_btn_tag: str
    image_name_input_tag: str
    take_picture_btn_tag: str
    highlight_theme: int
    last_group_name: str
    _sequence_started: bool
    protocols: list[str]

    def __init__(self,
                 label: str = "TARA Interface",
                 win_width: int = 350,
                 win_height: int = 350,
                 pos: tuple[int, int] = (10, 10),
                 outputs: dict[str, Any] | None = None,
                 uuid: str | None = None,
                 visible: bool = True,
                 **kwargs: Any) -> None:

        # Load OD50 config
        config_path = os.path.join(os.path.dirname(__file__), "od50_config.json")
        try:
            with open(config_path, "r") as f:
                raw_config = json.load(f)
            # Convert old format to new format if needed
            self.od50_config = {}
            for k, v in raw_config.items():
                if isinstance(v, dict):
                    self.od50_config[k] = v
                else:
                    # Old format: simple value, convert to new format
                    self.od50_config[k] = {"default": v, "keep": True}
        except Exception as e:
            logger.error(f"Failed to load OD50 config: {e}")
            # Fallback to hardcoded keys if file missing
            self.od50_config = {k: {"default": None, "keep": True} for k in ["Project", "Date", "Country", "Latitude", "Longitude", "Site", "Species", "Genotype", "Condition", "Temperature", "Timepoint"]}

        # OD50 Nomenclature fields
        self.naming_fields_keys = list(self.od50_config.keys())
        # Store field config for easy access
        self.field_config = self.od50_config
        self._persistent_fields = ["label", "target_signal", "protocol_step", "pipeline_mode", "naming_values"]

        if outputs is None:
            outputs = {
                "Fluo CMD": IOTypes.CMD_DICT,
                "VBA CMD": IOTypes.CMD_DICT,
                "Metadata": IOTypes.CMD_DICT,
                "Protocol File": IOTypes.TEXT,
                "Folder Fluo": IOTypes.FOLDER_PATH,
                "Folder VBA": IOTypes.FOLDER_PATH,
                "Frame": IOTypes.FRAME12,
            }

        super().__init__(label=label, pos=pos, win_width=win_width,
                        win_height=win_height, uuid=uuid,
                        outputs=outputs, visible=visible, **kwargs)
        
        # Merge saved values over JSON defaults
        defaults = {}
        for k, config in self.od50_config.items():
            default_val = config.get("default", "")
            if default_val is None:
                default_val = ""
            defaults[k] = default_val
        
        saved_values = getattr(self, "naming_values", {})
        self.naming_values = defaults.copy()
        if isinstance(saved_values, dict):
            # Only keep values for keys that exist in current config and are not empty
            for k, v in saved_values.items():
                if k in self.naming_fields_keys and v:
                    self.naming_values[k] = v

        # Auto-complete Date if it's empty and present in config
        if "Date" in self.naming_fields_keys and not self.naming_values.get("Date"):
            self.naming_values["Date"] = datetime.date.today().strftime("%Y%m%d")

        self.target_signal = getattr(self, "target_signal", 50.0)
        self.protocol_step = getattr(self, "protocol_step", 0)
        self.pipeline_mode = getattr(self, "pipeline_mode", None)  # "capture" | "processing" | None
        self.current_folder = None
        self.master_folder = None
        self.working_folder = None
        self.last_date_folder = None
        self._protocol_set_clicked = False
        self._metadata_set_clicked = False
        self._subfolders_created = False
        self._test_detection_clicked = False
        self._picture_taken = False
        self._sequence_started = False
        self._sequence_running = False
        self._sequence_stopped = False
        
        # Initialize master folder from config working directory
        try:
            from core.config_manager import config
            working_dir = config.get("Paths", {}).get("working_directory", "experiments")
            if os.path.isabs(working_dir):
                self.master_folder = working_dir
            else:
                # Relative path - resolve from repository root
                repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                self.master_folder = os.path.join(repo_root, working_dir)
        except Exception as e:
            logger.warning(f"Could not load working directory from config: {e}")
            self.master_folder = os.getcwd()

        # IO types
        self.accepted_input_types = [IOTypes.SAMPLE, IOTypes.FOLDER_PATH, IOTypes.FRAME12, IOTypes.FRAME, IOTypes.TRIGGER]
        self.output_types = [IOTypes.CMD_DICT, IOTypes.TEXT, IOTypes.FOLDER_PATH]

        # Find best exposure – frame counter state
        self._find_exp_active = False          # True while waiting for frames
        self._find_exp_frame_count = 0         # frames received so far
        self._find_exp_expected_frames = 0     # total frames expected
        self._progress_popup_tag    = f"tara_progress_popup_{self.UUID}"
        self._progress_bar_tag      = f"tara_progress_bar_{self.UUID}"
        self._progress_text_tag     = f"tara_progress_text_{self.UUID}"

        self.outputs = outputs or {}
        self.connections = {k: [] for k in self.outputs}

        # UI tags
        self.start_protocol_btn_tag = f"tara_start_proto_{self.UUID}"
        self.step1_group_tag = f"tara_s1_{self.UUID}"
        self.step2_group_tag = f"tara_s2_{self.UUID}"
        self.step3_group_tag = f"tara_s3_{self.UUID}"
        self.step4_group_tag = f"tara_s4_{self.UUID}"
        self.step5_group_tag = f"tara_s5_{self.UUID}"
        self.step6_group_tag = f"tara_s6_{self.UUID}"
        self.step8_group_tag = f"tara_s8_{self.UUID}"
        self.step9_group_tag = f"tara_s9_{self.UUID}"
        self.unified_next_btn_tag = f"tara_unified_next_{self.UUID}"
        self.unified_prev_btn_tag = f"tara_unified_prev_{self.UUID}"
        self.skip_vba_btn_tag = f"tara_skip_vba_{self.UUID}"

        self.select_folder_btn_tag = f"tara_sel_folder_{self.UUID}"
        self.target_signal_tag = f"tara_target_signal_{self.UUID}"
        self.find_best_exp_btn_tag = f"tara_find_best_exp_{self.UUID}"
        self.best_exp_int_tag = f"tara_best_exp_int_{self.UUID}"
        self.set_expos_btn_tag = f"tara_set_exp_{self.UUID}"
        self.protocol_combo_tag = f"tara_protocol_combo_{self.UUID}"
        self.run_protocol_btn_tag = f"tara_run_protocol_{self.UUID}"
        self.group_name_input_tag = f"tara_group_name_{self.UUID}"
        self.naming_toggle_tag = f"tara_naming_toggle_{self.UUID}"
        self.complex_naming_group_tag = f"tara_naming_group_{self.UUID}"
        self.simple_naming_group_tag = f"tara_simple_naming_group_{self.UUID}"
        self.naming_fields_tags = {f: f"tara_f_{f}_{self.UUID}" for f in self.naming_fields_keys}
        self.naming_fields_custom_tags = {f: f"tara_f_{f}_custom_{self.UUID}" for f in self.naming_fields_keys}
        self.naming_fields_combo_tags = {f: f"tara_f_{f}_combo_{self.UUID}" for f in self.naming_fields_keys}
        self.protocol_tooltip_text_tag = f"tara_protocol_tt_{self.UUID}"
        self.save_metadata_btn_tag = f"tara_save_meta_{self.UUID}"
        self.start_sequence_btn_tag = f"tara_start_seq_{self.UUID}"
        
        # Subfolder structure tags
        self.main_subfolder_tag = f"tara_main_subfolder_{self.UUID}"
        self.fvfm_subfolder_tag = f"tara_fvfm_subfolder_{self.UUID}"
        self.superseq_subfolder_tag = f"tara_superseq_subfolder_{self.UUID}"
        self.create_subfolders_btn_tag = f"tara_create_subfolders_{self.UUID}"
        
        # Master folder and working folder tags
        self.master_folder_btn_tag = f"tara_master_folder_{self.UUID}"
        self.master_folder_label_tag = f"tara_master_folder_label_{self.UUID}"
        self.working_folder_btn_tag = f"tara_working_folder_{self.UUID}"
        self.working_folder_label_tag = f"tara_working_folder_label_{self.UUID}"

        # ── Pipeline mode selection tags ──────────────────────────────────────
        self.mode_select_group_tag = f"tara_mode_select_{self.UUID}"
        self.capture_mode_btn_tag = f"tara_capture_mode_{self.UUID}"
        self.processing_mode_btn_tag = f"tara_processing_mode_{self.UUID}"
        self.processing_vba_mode_btn_tag = f"tara_processing_vba_mode_{self.UUID}"

        # ── Processing pipeline: step-1 folder selection ──────────────────────
        self.proc_step1_group_tag = f"tara_proc_s1_{self.UUID}"
        self.proc_working_folder_btn_tag = f"tara_proc_wfolder_{self.UUID}"
        self.proc_working_folder_label_tag = f"tara_proc_wfolder_label_{self.UUID}"

        # ── Capture pipeline: done/success screen ─────────────────────────────
        self.capture_done_group_tag = f"tara_capture_done_{self.UUID}"
        self.switch_to_processing_btn_tag = f"tara_switch_proc_{self.UUID}"

        # ── Set detector to 0 (TARA) for initial startup ─────────────────────
        from core.app_state import app_state
        app_state.detector = 0
        
        # Build UI
        with dpg.window(label=self.label, width=self.win_width, height=self.win_height,
                        pos=self.pos, tag=self.winID, show=self.visible):
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Main menu", width=-1, callback=self._reset_experiment)
            dpg.add_separator()
            
            dpg.add_button(label="Start Protocol", tag=self.start_protocol_btn_tag, width=-1, height=40,
                           show=False, callback=lambda: self._set_mode("capture"))

            # ── Mode selection screen ─────────────────────────────────────────
            with dpg.group(tag=self.mode_select_group_tag, show=True):
                dpg.add_text("Choose your pipeline:", color=(200, 200, 200))
                dpg.add_spacer(height=6)
                dpg.add_button(label=u"\uf030  Data Capture",
                               tag=self.capture_mode_btn_tag,
                               width=-1, height=42,
                               callback=lambda: self._set_mode("capture"))
                dpg.add_spacer(height=8)
                dpg.add_button(label=u"\uf080  Data Processing (Fluo)",
                               tag=self.processing_mode_btn_tag,
                               width=-1, height=42,
                               callback=lambda: self._set_mode("processing"))
                dpg.add_spacer(height=8)
                dpg.add_button(label=u"\uf080   Data Processing (VBA)",
                               tag=self.processing_vba_mode_btn_tag,
                               width=-1, height=42,
                               callback=lambda: self._set_mode("processing_vba"))
            
            dpg.add_spacer(height=6)
            dpg.add_button(label="\uf060 Prev", tag=self.unified_prev_btn_tag, width=-1, show=False, callback=self._on_unified_prev)
            dpg.add_spacer(height=4)

            with dpg.group(tag=self.step1_group_tag, show=False):
                self._add_step_header(1, "Select Master Folder")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Select Master Folder", tag=self.master_folder_btn_tag, callback=self._on_master_folder_clicked)
                    dpg.add_text("None", tag=self.master_folder_label_tag, color=(150, 150, 150))
                
                dpg.add_spacer(height=4)
                dpg.add_separator()
                dpg.add_text("Create Date Folder Structure", color=(100, 200, 255))
                dpg.add_spacer(height=2)
                dpg.add_input_text(label="Date (YYYYMMDD)", tag=self.main_subfolder_tag, default_value=datetime.date.today().strftime("%Y%m%d"))
                dpg.add_spacer(height=4)
                dpg.add_button(label="Create Date Folder (FvFm + SuperSeq)", tag=self.create_subfolders_btn_tag, callback=self._on_create_subfolders_clicked, width=-1)
                
                dpg.add_spacer(height=4)
                dpg.add_separator()
                self._add_step_header(1.1, "Select Working Folder")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Select Working Folder", tag=self.working_folder_btn_tag, callback=self._on_working_folder_clicked)
                    dpg.add_text("None", tag=self.working_folder_label_tag, color=(150, 150, 150))
                
                dpg.add_separator()

            with dpg.group(tag=self.step2_group_tag, show=False):
                self._add_step_header(2, "Auto Config Expos")
                with dpg.group(horizontal=True):
                    dpg.add_text("Target Signal:")
                    dpg.add_input_float(tag=self.target_signal_tag, default_value=self.target_signal, 
                                      width=100, step=-1, callback=self._update_target_signal)
                
                    dpg.add_button(label="Find best expos", width=-1,tag=self.find_best_exp_btn_tag,
                                callback=self._send_find_exp_cmd)
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Best Exp (µs) :")
                    dpg.add_input_int(tag=self.best_exp_int_tag, default_value=0, width=100, step=0, callback=self._send_set_expos_cmd)

                dpg.add_separator()

            with dpg.group(tag=self.step3_group_tag, show=False):
                self._add_step_header(3, "Test Detection")
                self.test_det_btn_tag = f"tara_testdet_btn_{self.UUID}"
                dpg.add_button(label="Test Detection", tag=self.test_det_btn_tag, width=-1, callback=self._send_test_detection_cmd)
                dpg.add_separator()

            with dpg.group(tag=self.step4_group_tag, show=False):
                self._add_step_header(4, "Select Protocol")
                
                self.protocols = self._get_protocols()
                
                dpg.add_combo(self.protocols, tag=self.protocol_combo_tag, width=-1, callback=self._update_protocol_tooltip)
                
                with dpg.tooltip(parent=self.protocol_combo_tag):
                    dpg.add_text("Select a protocol run", tag=self.protocol_tooltip_text_tag)
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Sample name / Group:")
                    dpg.add_input_text(hint="e.g. TestSequence1", tag=self.group_name_input_tag, callback=self._on_protocol_input_changed)

                dpg.add_button(label="Set Protocol", tag=self.run_protocol_btn_tag, width=-1,
                                callback=self._send_protocol_cmd)

            with dpg.group(tag=self.step5_group_tag, show=False):
                self._add_step_header(5, "Run Sequence")
                dpg.add_button(label="Start sequence", tag=self.start_sequence_btn_tag, width=-1, height=35, callback=self._send_start_sequence_cmd)
                dpg.add_separator()

            with dpg.group(tag=self.step8_group_tag, show=False):
                self._add_step_header(8, "Position Sample")
                dpg.add_text("Position the sample under the camera, then click Next.", color=(180, 180, 180))
                dpg.add_separator()

            with dpg.group(tag=self.step6_group_tag, show=False):
                self._add_step_header(6, "Provide Metadata (OD50)")
                
                with dpg.group(tag=self.complex_naming_group_tag):
                    for f in self.naming_fields_keys:
                        field_config = self.field_config.get(f, {})
                        if not field_config.get("show_ui", True):
                            continue
                        options = field_config.get("options", None)
                        
                        with dpg.group(horizontal=True):
                            dpg.add_text(f"{f}:")
                            
                            if options:
                                # Create dropdown with options
                                combo_options = options + ["Custom"]
                                default_value = self.naming_values.get(f, "")
                                if default_value not in combo_options:
                                    default_value = "Custom"
                                
                                dpg.add_combo(combo_options, tag=self.naming_fields_combo_tags[f],
                                            default_value=default_value,
                                            width=-1,
                                            callback=self._on_combo_changed,
                                            user_data=f)
                            else:
                                # Regular text input
                                dpg.add_input_text(tag=self.naming_fields_tags[f], 
                                                default_value=self.naming_values.get(f, ""),
                                                width=-1,
                                                callback=self._update_naming_field,
                                                user_data=f)
                        
                        # Custom input field (separate vertical group, hidden by default)
                        if options:
                            dpg.add_input_text(tag=self.naming_fields_custom_tags[f],
                                            hint="Custom value...",
                                            default_value=self.naming_values.get(f, "") if default_value == "Custom" else "",
                                            width=-1,
                                            show=(default_value == "Custom"),
                                            callback=self._update_naming_field,
                                            user_data=f)
                
                dpg.add_button(label="Save Metadata", width=-1, tag=self.save_metadata_btn_tag, callback=self._send_metadata_cmd)
            with dpg.group(tag=self.step9_group_tag, show=False):
                self._add_step_header(9, "Analyze Results")
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Image Name:")
                    self.image_name_input_tag = f"tara_img_name_{self.UUID}"
                    dpg.add_input_text(tag=self.image_name_input_tag, width=150, callback=lambda sender, app_data: self._update_step_highlights())
                    
                self.take_picture_btn_tag = f"tara_take_picture_{self.UUID}"
                dpg.add_button(label="Take Picture", width=-1, tag=self.take_picture_btn_tag, callback=self._send_take_picture_cmd)

            # ── Processing pipeline – Step 1: Select Data Folder ──────────────────
            with dpg.group(tag=self.proc_step1_group_tag, show=False):
                with dpg.group(horizontal=True):
                    dpg.add_text("Step 1: Select Data Folder", color=(100, 200, 255))
                dpg.add_spacer(height=4)
                dpg.add_text("Select the folder that contains the raw images to process.",
                             color=(180, 180, 180))
                dpg.add_spacer(height=6)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Select Data Folder",
                                   tag=self.proc_working_folder_btn_tag,
                                   callback=self._on_proc_working_folder_clicked)
                    dpg.add_text("None", tag=self.proc_working_folder_label_tag, color=(150, 150, 150))
                dpg.add_separator()

            # ── Capture pipeline – Done screen ────────────────────────────────
            with dpg.group(tag=self.capture_done_group_tag, show=False):
                self.capture_done_prev_btn_tag = f"tara_done_prev_{self.UUID}"
                dpg.add_button(label="\uf060 Prev", tag=self.capture_done_prev_btn_tag, width=-1,
                               callback=lambda: self._on_step_prev(8), show=False)
                dpg.add_text("Acquisition Protocol Finished", color=(100, 200, 255))
                dpg.add_spacer(height=6)
                
                self.new_exp_btn_tag = f"tara_new_exp_{self.UUID}"
                dpg.add_button(label="Return to Main Menu", tag=self.new_exp_btn_tag, callback=self._reset_experiment, width=-1)
                
                self.close_software_btn_tag = f"tara_close_soft_{self.UUID}"
                dpg.add_button(label="Close Software", tag=self.close_software_btn_tag, callback=self._close_software, width=-1)
                dpg.add_separator()

            dpg.add_spacer(height=6)
            dpg.add_button(label="Next \uf061", tag=self.unified_next_btn_tag, height=35, width=-1, show=False, callback=self._on_unified_next)
            dpg.add_button(label="Skip to VBA \uf101", tag=self.skip_vba_btn_tag, height=30, width=-1, show=False, callback=self._on_skip_vba)

        # ── Hidden progress modal (shown/hidden, never recreated) ─────────────
        with dpg.window(
            label="Acquisition in progress…",
            tag=self._progress_popup_tag,
            modal=True,
            show=False,
            no_title_bar=True,
            no_resize=True,
            no_move=True,
            no_collapse=True,
            no_close=True,
            autosize=True,
            
        ):
            with dpg.group(horizontal=True):
                dpg.add_loading_indicator(radius=2.0)
                dpg.add_text("Finding best exposure…", tag=self._progress_text_tag)
            dpg.add_spacer(height=10)
            dpg.add_progress_bar(tag=self._progress_bar_tag, width=300)
            dpg.add_spacer(height=4)

        # Global keypress handler to allow manual frame skip in User Mode using DEL
        with dpg.handler_registry():
            dpg.add_key_press_handler(key=dpg.mvKey_Delete, callback=self._on_key_delete)

        # Apply initial state
        self._set_step(self.protocol_step)
        self.update_permission()
        
        # Set initial master folder label
        if self.master_folder and dpg.does_item_exist(self.master_folder_label_tag):
            dpg.set_value(self.master_folder_label_tag, f".../{os.path.basename(self.master_folder)}")

    def update_permission(self) -> None:
        """
        Adjust module permissions and UI visibility based on app_state.mode.
        Disable closing in user mode.
        """
        from core.app_state import app_state
        mode = app_state.mode
        is_user = (mode == "user")
        enabled = not is_user
        
        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def _get_tutorial_path(self, step_num: int | float) -> str:
        """Return the appropriate tutorial JSON path for a step.

        Steps 6 (Metadata) uses a mode-specific file when in processing pipelines.
        """
        base_dir = os.path.dirname(__file__)
        if step_num == 6 and self.pipeline_mode == "processing":
            return os.path.join(base_dir, "step6_fluo.json")
        if step_num == 6 and self.pipeline_mode == "processing_vba":
            return os.path.join(base_dir, "step6_vba.json")
        return os.path.join(base_dir, f"step{step_num}.json")

    def _play_step_tutorial(self, step_num: int | float) -> None:
        """Load and play the tutorial for the given step (mode-aware)."""
        from core.tutorial_manager import tutorial_manager
        json_path = self._get_tutorial_path(step_num)
        # Fallback to generic step file if mode-specific one does not exist
        if not os.path.exists(json_path):
            fallback = os.path.join(os.path.dirname(__file__), f"step{step_num}.json")
            if os.path.exists(fallback):
                json_path = fallback
            else:
                return
        tutorial_manager.load_and_play(json_path)

    def _has_any_tutorial(self, step_num: int | float) -> bool:
        """Return True if any tutorial JSON (generic or mode-specific) exists."""
        base_dir = os.path.dirname(__file__)
        candidates = [os.path.join(base_dir, f"step{step_num}.json")]
        if step_num == 6:
            candidates += [
                os.path.join(base_dir, "step6_fluo.json"),
                os.path.join(base_dir, "step6_vba.json"),
            ]
        return any(os.path.exists(p) for p in candidates)

    def _add_step_header(self, step_num: int | float, text: str) -> None:
        """Adds a step title with an optional Help button if a tutorial JSON exists."""
        has_tutorial = self._has_any_tutorial(step_num)

        with dpg.group(horizontal=True):
            dpg.add_text(f"Step {step_num}: {text}", color=(100, 200, 255))
            if has_tutorial:
                dpg.add_button(label="\uf059",
                               callback=lambda s, a, u: self._play_step_tutorial(step_num))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text(f"Click to start the tutorial for Step {step_num}")

    def _update_protocol_tooltip(self, sender: Any, app_data: Any, *args: Any, **kwargs: Any) -> None:
        self._protocol_set_clicked = False
        selected = app_data
        if not selected:
            self._update_step_highlights()
            return
            
        import os
        base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "sequence_processor", "sequences")
        file_path = os.path.join(base_dir, selected)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if dpg.does_item_exist(self.protocol_tooltip_text_tag):
                dpg.set_value(self.protocol_tooltip_text_tag, content)
        except Exception as e:
            if dpg.does_item_exist(self.protocol_tooltip_text_tag):
                dpg.set_value(self.protocol_tooltip_text_tag, f"Error reading file: {e}")
                
        self._update_step_highlights()

    def _get_protocols(self) -> list[str]:
        import os
        base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "sequence_processor", "sequences")
        protocols = []
        if os.path.exists(base_dir):
            for root, _, files in os.walk(base_dir):
                for f in files:
                    if f.endswith('.txt'):
                        rel_path = os.path.relpath(os.path.join(root, f), base_dir)
                        protocols.append(rel_path)
        return sorted(protocols)

    def _reset_experiment(self, *args: Any, **kwargs: Any) -> None:
        self.current_folder = None
        self.master_folder = None
        self.working_folder = None
        self.pipeline_mode = None
        self._protocol_set_clicked = False
        self._sequence_started = False
        self._sequence_running = False
        self._sequence_stopped = False
        self._test_detection_clicked = False
        self._picture_taken = False
        
        # Reinitialize master folder from config working directory
        try:
            from core.config_manager import config
            working_dir = config.get("Paths", {}).get("working_directory", "experiments")
            if os.path.isabs(working_dir):
                self.master_folder = working_dir
            else:
                # Relative path - resolve from repository root
                repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                self.master_folder = os.path.join(repo_root, working_dir)
        except Exception as e:
            logger.warning(f"Could not load working directory from config: {e}")
            self.master_folder = os.getcwd()
        
        reset_cmd = {
            "clear": True,
            "save": False,
            "record": False,
            "add_timestamp": False
        }
        for k in ["Folder Fluo", "Folder VBA"]:
            for connected_module in self.connections.get(k, []):
                try:
                    connected_module.input_cb(data=reset_cmd, data_type=IOTypes.CMD_DICT)
                except Exception:
                    pass

        for connected_module in self.connections.get("Metadata", []):
            try:
                connected_module.input_cb(data={"clear": True}, data_type=IOTypes.CMD_DICT)
            except Exception:
                pass

        self._emit_fluo_cmd({"trigger_mode": "Video"})
        self._emit_vba_cmd({"trigger_mode": "Video"})
        self._emit_vba_cmd({"enabled": False})
        self._emit_fluo_cmd({"enabled": False})
        self._emit_fluo_cmd({"white": 0})
        
        # Reset folder labels
        if dpg.does_item_exist(self.master_folder_label_tag):
            dpg.set_value(self.master_folder_label_tag, f".../{os.path.basename(self.master_folder)}")
        if dpg.does_item_exist(self.working_folder_label_tag):
            dpg.set_value(self.working_folder_label_tag, "None")
        if dpg.does_item_exist(getattr(self, "folder_label_tag", "")):
            dpg.set_value(self.folder_label_tag, "None")
        if dpg.does_item_exist(getattr(self, "proc_working_folder_label_tag", "")):
            dpg.set_value(self.proc_working_folder_label_tag, "None")
            
        if dpg.does_item_exist(self.best_exp_int_tag):
            dpg.set_value(self.best_exp_int_tag, 0)
            
        if dpg.does_item_exist(getattr(self, "group_name_input_tag", "")):
            dpg.set_value(self.group_name_input_tag, "")

        for f in self.naming_fields_keys:
            field_config = self.field_config.get(f, {})
            default_val = field_config.get("default", "")
            if default_val is None:
                default_val = ""
            
            # Auto-complete Date if it's empty
            if f == "Date" and not default_val:
                default_val = datetime.date.today().strftime("%Y%m%d")

            options = field_config.get("options", None)
            if options:
                # Handle dropdown
                combo_tag = self.naming_fields_combo_tags[f]
                custom_tag = self.naming_fields_custom_tags[f]
                
                # Set dropdown value
                if dpg.does_item_exist(combo_tag):
                    if default_val in options:
                        dpg.set_value(combo_tag, default_val)
                        if dpg.does_item_exist(custom_tag):
                            dpg.configure_item(custom_tag, show=False)
                    else:
                        dpg.set_value(combo_tag, "Custom")
                        if dpg.does_item_exist(custom_tag):
                            dpg.set_value(custom_tag, default_val)
                            dpg.configure_item(custom_tag, show=True)
                
                self.naming_values[f] = default_val
            else:
                # Handle regular text input
                if dpg.does_item_exist(self.naming_fields_tags[f]):
                    dpg.set_value(self.naming_fields_tags[f], default_val)
                self.naming_values[f] = default_val
            
        # Reset subfolder config values
        if dpg.does_item_exist(self.main_subfolder_tag):
            dpg.set_value(self.main_subfolder_tag, datetime.date.today().strftime("%Y%m%d"))
        if dpg.does_item_exist(self.fvfm_subfolder_tag):
            dpg.set_value(self.fvfm_subfolder_tag, "FvFm")
        if dpg.does_item_exist(self.superseq_subfolder_tag):
            dpg.set_value(self.superseq_subfolder_tag, "SuperSeq")

        self._set_step(0)
        logger.info("TARA experiment state reset to fresh.")
        
        from core.module_registry import apply_named_view
        apply_named_view("new_exp")

    def _close_software(self, *args: Any, **kwargs: Any) -> None:
        """Turn off all fluo controller outputs and close the application."""
        logger.info("Closing software - turning off all hardware outputs")

        # Send emergency stop command to turn off actinic, detector, and frame rate
        self._emit_fluo_cmd({"send": "act 0 \n det 0 \n fr 0\n"})

        # Turn off white lights
        self._emit_fluo_cmd({"white": 0})
        self._emit_fluo_cmd({"white2": 0})

        # Disconnect fluo controller if connected
        for connected_module in self.connections.get("Fluo CMD", []):
            try:
                connected_module.input_cb(data={"disconnect": True}, data_type=IOTypes.CMD_DICT)
            except Exception:
                pass

        logger.info("Hardware outputs turned off. Closing application...")

        from core.app_state import app_state
        app_state.close_requested = True

    def _set_step(self, step: int) -> None:
        self.protocol_step = step
        if not dpg.does_item_exist(self.winID):
            return

        # Always hide the legacy Start Protocol button
        dpg.configure_item(self.start_protocol_btn_tag, show=False)

        # Mode selection screen shown only at step 0 (no mode chosen)
        if dpg.does_item_exist(getattr(self, "mode_select_group_tag", "")):
            dpg.configure_item(self.mode_select_group_tag, show=(step == 0))

        if self.pipeline_mode == "capture":
            # ── Capture: show ONLY the current step ───────────────────
            dpg.configure_item(self.step1_group_tag, show=(step == 1))
            dpg.configure_item(self.step2_group_tag, show=(step == 2))
            dpg.configure_item(self.step3_group_tag, show=(step == 3))
            if dpg.does_item_exist(getattr(self, "step4_group_tag", "")):
                dpg.configure_item(self.step4_group_tag, show=(step == 4))
            if dpg.does_item_exist(getattr(self, "step5_group_tag", "")):
                dpg.configure_item(self.step5_group_tag, show=(step == 5))
            # Step 6 = Position Sample  (reuses step8_group_tag)
            if dpg.does_item_exist(getattr(self, "step8_group_tag", "")):
                dpg.configure_item(self.step8_group_tag, show=(step == 6))
            # Step 7 = VBA Analysis     (reuses step9_group_tag)
            if dpg.does_item_exist(getattr(self, "step9_group_tag", "")):
                dpg.configure_item(self.step9_group_tag, show=(step == 7))
            if dpg.does_item_exist(getattr(self, "capture_done_group_tag", "")):
                dpg.configure_item(self.capture_done_group_tag, show=(step == 8))
            # Hide processing-only groups
            for attr in ["proc_step1_group_tag", "step6_group_tag"]:
                tag = getattr(self, attr, "")
                if dpg.does_item_exist(tag):
                    dpg.configure_item(tag, show=False)
            # Next visible for steps 1–7
            if dpg.does_item_exist(getattr(self, "unified_next_btn_tag", "")):
                dpg.configure_item(self.unified_next_btn_tag, show=(0 < step < 8))
            # Unified Prev visible for steps 2–8
            if dpg.does_item_exist(getattr(self, "unified_prev_btn_tag", "")):
                dpg.configure_item(self.unified_prev_btn_tag, show=(1 < step <= 8))

        elif self.pipeline_mode == "processing":
            # ── Processing: show ONLY the current step ───────────────
            for tag in [self.step1_group_tag, self.step2_group_tag, self.step3_group_tag]:
                dpg.configure_item(tag, show=False)
            for attr in ["step4_group_tag", "step5_group_tag", "capture_done_group_tag",
                        "step8_group_tag", "step9_group_tag"]:
                tag = getattr(self, attr, "")
                if dpg.does_item_exist(tag):
                    dpg.configure_item(tag, show=False)
            if dpg.does_item_exist(getattr(self, "proc_step1_group_tag", "")):
                dpg.configure_item(self.proc_step1_group_tag, show=(step == 1))
            if dpg.does_item_exist(getattr(self, "step6_group_tag", "")):
                dpg.configure_item(self.step6_group_tag, show=(step == 2))
            # Next visible for step 1 only
            if dpg.does_item_exist(getattr(self, "unified_next_btn_tag", "")):
                dpg.configure_item(self.unified_next_btn_tag, show=(step == 1))
            # Unified Prev visible for all processing steps
            if dpg.does_item_exist(getattr(self, "unified_prev_btn_tag", "")):
                dpg.configure_item(self.unified_prev_btn_tag, show=(step >= 1))

        elif self.pipeline_mode == "processing_vba":
            # ── Processing VBA: show ONLY the current step ────────────
            for tag in [self.step1_group_tag, self.step2_group_tag, self.step3_group_tag]:
                dpg.configure_item(tag, show=False)
            for attr in ["step4_group_tag", "step5_group_tag",
                          "step8_group_tag", "step9_group_tag", "capture_done_group_tag"]:
                tag = getattr(self, attr, "")
                if dpg.does_item_exist(tag):
                    dpg.configure_item(tag, show=False)
            
            if dpg.does_item_exist(getattr(self, "proc_step1_group_tag", "")):
                dpg.configure_item(self.proc_step1_group_tag, show=(step == 1))
            if dpg.does_item_exist(getattr(self, "step6_group_tag", "")):
                dpg.configure_item(self.step6_group_tag, show=(step == 2))

            # Next visible for step 1 only
            if dpg.does_item_exist(getattr(self, "unified_next_btn_tag", "")):
                dpg.configure_item(self.unified_next_btn_tag, show=(step == 1))

            # Unified Prev visible for all processing steps
            if dpg.does_item_exist(getattr(self, "unified_prev_btn_tag", "")):
                dpg.configure_item(self.unified_prev_btn_tag, show=(step >= 1))

        else:
            # ── No mode selected: hide everything ─────────────────────────────
            for tag in [self.step1_group_tag, self.step2_group_tag, self.step3_group_tag]:
                dpg.configure_item(tag, show=False)
            for attr in ["step4_group_tag", "step5_group_tag", "step6_group_tag",
                         "step8_group_tag", "step9_group_tag",
                         "proc_step1_group_tag", "capture_done_group_tag"]:
                tag = getattr(self, attr, "")
                if dpg.does_item_exist(tag):
                    dpg.configure_item(tag, show=False)
            if dpg.does_item_exist(getattr(self, "unified_next_btn_tag", "")):
                dpg.configure_item(self.unified_next_btn_tag, show=False)
            if dpg.does_item_exist(getattr(self, "unified_prev_btn_tag", "")):
                dpg.configure_item(self.unified_prev_btn_tag, show=False)

        # Skip VBA button: hidden in new two-pipeline flow
        if dpg.does_item_exist(getattr(self, "skip_vba_btn_tag", "")):
            dpg.configure_item(self.skip_vba_btn_tag, show=False)

        # Autoscroll to the unified Prev button so the current step is at top
        if step > 0:
            _scroll_tag = getattr(self, "unified_prev_btn_tag", "")

            def _scroll(tag: str = _scroll_tag) -> None:
                import time
                time.sleep(0.05)   # ~1 frame at 20 fps minimum
                if not dpg.does_item_exist(self.winID):
                    return
                if tag and dpg.does_item_exist(tag):
                    try:
                        win_pos_y = dpg.get_item_pos(self.winID)[1]
                        item_y = dpg.get_item_rect_min(tag)[1]
                        current_scroll = dpg.get_y_scroll(self.winID)
                        dpg.set_y_scroll(self.winID, max(0, current_scroll + item_y - win_pos_y - 24))
                    except Exception:
                        dpg.set_y_scroll(self.winID, dpg.get_y_scroll_max(self.winID))
                else:
                    dpg.set_y_scroll(self.winID, dpg.get_y_scroll_max(self.winID))
            import threading
            threading.Thread(target=_scroll, daemon=True).start()

        # Update button highlights for step guidance
        self._update_step_highlights()

    def _update_step_highlights(self) -> None:
        """Dynamically update theme highlights for buttons and inputs based on step progress."""
        if not dpg.does_item_exist(self.winID):
            return

        # Ensure our custom highlight theme is created with Orange Accent
        if not hasattr(self, "highlight_theme") or not dpg.does_item_exist(self.highlight_theme):
            self.highlight_theme = dpg.add_theme()
            # Orange theme for active button
            with dpg.theme_component(dpg.mvButton, parent=self.highlight_theme):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (255, 130, 0, 255))        # Vibrant Orange
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 155, 30, 255)) # Hover Orange
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (220, 100, 0, 255))   # Active Orange
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4.0)
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 2.0)
                dpg.add_theme_color(dpg.mvThemeCol_Border, (255, 200, 50, 255))        # Golden Border

            # Accent highlight theme for Combo dropdown
            with dpg.theme_component(dpg.mvCombo, parent=self.highlight_theme):
                dpg.add_theme_color(dpg.mvThemeCol_Border, (255, 130, 0, 255))         # Orange Border
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 2.0)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (50, 40, 30, 255))         # Warm background

            # Accent highlight theme for InputText box
            with dpg.theme_component(dpg.mvInputText, parent=self.highlight_theme):
                dpg.add_theme_color(dpg.mvThemeCol_Border, (255, 130, 0, 255))         # Orange Border
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 2.0)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (50, 40, 30, 255))         # Warm background

        step = self.protocol_step
        mode = self.pipeline_mode

        # Reset all buttons/inputs to default theme first
        all_buttons = [
            getattr(self, "working_folder_btn_tag", ""),
            getattr(self, "find_best_exp_btn_tag", ""),
            getattr(self, "test_det_btn_tag", ""),
            getattr(self, "protocol_combo_tag", ""),
            getattr(self, "group_name_input_tag", ""),
            getattr(self, "run_protocol_btn_tag", ""),
            getattr(self, "start_sequence_btn_tag", ""),
            getattr(self, "image_name_input_tag", ""),
            getattr(self, "take_picture_btn_tag", ""),
            getattr(self, "proc_working_folder_btn_tag", ""),
            getattr(self, "save_metadata_btn_tag", ""),
            getattr(self, "unified_next_btn_tag", "")
        ]
        for btn in all_buttons:
            if btn and dpg.does_item_exist(btn):
                dpg.bind_item_theme(btn, 0)

        # Decide which button/input to highlight
        target_btn = None
        action_done = False

        if mode == "capture":
            if step == 1:
                target_btn = getattr(self, "working_folder_btn_tag", None)
                action_done = bool(self.working_folder and os.path.exists(self.working_folder))
            elif step == 2:
                target_btn = getattr(self, "find_best_exp_btn_tag", None)
                best_exp = dpg.get_value(self.best_exp_int_tag) if dpg.does_item_exist(self.best_exp_int_tag) else 0
                action_done = (best_exp > 0)
            elif step == 3:
                target_btn = getattr(self, "test_det_btn_tag", None)
                action_done = getattr(self, "_test_detection_clicked", False)
            elif step == 4:
                selected_protocol = dpg.get_value(self.protocol_combo_tag) if dpg.does_item_exist(self.protocol_combo_tag) else ""
                group_name = dpg.get_value(self.group_name_input_tag) if dpg.does_item_exist(self.group_name_input_tag) else ""
                
                if not selected_protocol:
                    target_btn = getattr(self, "protocol_combo_tag", None)
                    action_done = False
                elif not group_name.strip():
                    target_btn = getattr(self, "group_name_input_tag", None)
                    action_done = False
                else:
                    target_btn = getattr(self, "run_protocol_btn_tag", None)
                    action_done = getattr(self, "_protocol_set_clicked", False)
            elif step == 5:
                target_btn = getattr(self, "start_sequence_btn_tag", None)
                action_done = getattr(self, "_sequence_started", False)
            elif step == 6:
                # No specific button action in Position Sample step; Next is always highlighted
                target_btn = None
                action_done = True
            elif step == 7:
                img_name = dpg.get_value(self.image_name_input_tag) if dpg.does_item_exist(self.image_name_input_tag) else ""
                if not img_name.strip():
                    target_btn = getattr(self, "image_name_input_tag", None)
                    action_done = False
                else:
                    target_btn = getattr(self, "take_picture_btn_tag", None)
                    action_done = getattr(self, "_picture_taken", False)

        elif mode == "processing":
            if step == 1:
                target_btn = getattr(self, "proc_working_folder_btn_tag", None)
                action_done = bool(self.working_folder and os.path.exists(self.working_folder))
            elif step == 2:
                target_btn = getattr(self, "save_metadata_btn_tag", None)
                action_done = getattr(self, "_metadata_set_clicked", False)

        elif mode == "processing_vba":
            if step == 1:
                target_btn = getattr(self, "proc_working_folder_btn_tag", None)
                action_done = bool(self.working_folder and os.path.exists(self.working_folder))
            elif step == 2:
                target_btn = getattr(self, "save_metadata_btn_tag", None)
                action_done = getattr(self, "_metadata_set_clicked", False)

        # Apply highlight theme
        if target_btn and dpg.does_item_exist(target_btn) and not action_done:
            dpg.bind_item_theme(target_btn, self.highlight_theme)
        elif action_done:
            # If the primary action is done, highlight the Next button (if shown)
            next_btn = getattr(self, "unified_next_btn_tag", None)
            if next_btn and dpg.does_item_exist(next_btn):
                # Only highlight next if it is actually visible for this step
                is_visible = dpg.get_item_configuration(next_btn).get("show", False)
                if is_visible:
                    dpg.bind_item_theme(next_btn, self.highlight_theme)
        
    def _on_unified_prev(self, *args: Any, **kwargs: Any) -> None:
        """Dispatch Prev button to the active pipeline."""
        if self.pipeline_mode == "capture":
            self._on_step_prev(self.protocol_step)
        elif self.pipeline_mode == "processing":
            if self.protocol_step == 1:
                self._on_proc_step1_prev()
            else:
                self._on_proc_prev_from_ui(6)
        elif self.pipeline_mode == "processing_vba":
            if self.protocol_step == 1:
                self._on_proc_step1_prev()
            else:
                self._on_proc_vba_prev_from_ui(6)

    def _on_unified_next(self, *args: Any, **kwargs: Any) -> None:
        """Dispatch Next button to the active pipeline."""
        if self.pipeline_mode == "capture":
            self._on_capture_next(self.protocol_step)
        elif self.pipeline_mode == "processing":
            self._on_proc_next(self.protocol_step)
        elif self.pipeline_mode == "processing_vba":
            self._on_proc_vba_next(self.protocol_step)

    def _on_step_prev(self, current_step: int) -> None:
        # In processing mode delegate to the processing-specific handler
        if self.pipeline_mode == "processing":
            self._on_proc_prev_from_ui(current_step)
            return

        if self.pipeline_mode == "processing_vba":
            self._on_proc_vba_prev_from_ui(current_step)
            return

        # ── Capture mode ──────────────────────────────────────────────
        # Clear current step AND the step we are going back to
        steps_to_clear = [current_step, current_step - 1]
        
        for step in steps_to_clear:
            if step == 1:
                self.current_folder = None
                self.working_folder = None
                self._subfolders_created = False
                if dpg.does_item_exist(self.working_folder_label_tag):
                    dpg.set_value(self.working_folder_label_tag, "None")
            elif step == 2:
                if dpg.does_item_exist(self.best_exp_int_tag):
                    dpg.set_value(self.best_exp_int_tag, 0)
            elif step == 3:
                self._test_detection_clicked = False
            elif step == 4:
                if dpg.does_item_exist(self.group_name_input_tag):
                    dpg.set_value(self.group_name_input_tag, "")
                self._protocol_set_clicked = False
            elif step == 5:
                self._sequence_started = False
                self._sequence_running = False
                self._sequence_stopped = False
            elif step == 6:
                self._metadata_set_clicked = False
                # Clear fields based on keep flag from config
                for f in self.naming_fields_keys:
                    field_config = self.field_config.get(f, {})
                    should_keep = field_config.get("keep", True)
                    
                    if not should_keep:
                        options = field_config.get("options", None)
                        if options:
                            # Handle dropdown
                            combo_tag = self.naming_fields_combo_tags[f]
                            custom_tag = self.naming_fields_custom_tags[f]
                            
                            # Reset to default
                            default_val = field_config.get("default", "")
                            if default_val is None:
                                default_val = ""
                            
                            if dpg.does_item_exist(combo_tag):
                                if default_val in options:
                                    dpg.set_value(combo_tag, default_val)
                                    if dpg.does_item_exist(custom_tag):
                                        dpg.configure_item(custom_tag, show=False)
                                else:
                                    dpg.set_value(combo_tag, "Custom")
                                    if dpg.does_item_exist(custom_tag):
                                        dpg.set_value(custom_tag, default_val)
                                        dpg.configure_item(custom_tag, show=True)
                            
                            self.naming_values[f] = default_val
                        else:
                            # Handle regular text input
                            if dpg.does_item_exist(self.naming_fields_tags[f]):
                                dpg.set_value(self.naming_fields_tags[f], "")
                            self.naming_values[f] = ""
            elif step == 7:
                self._picture_taken = False
            elif step == 9:
                if dpg.does_item_exist(getattr(self, "image_name_input_tag", "")):
                    dpg.set_value(self.image_name_input_tag, "")

        if self.protocol_step > 0:
            target_step = self.protocol_step - 1
            self._set_step(target_step)

            from core.module_registry import apply_named_view
            if target_step in [2, 3]:
                apply_named_view("calibration")
            elif target_step in [4, 5]:
                apply_named_view("sequence")
            elif target_step == 6:
                apply_named_view("VBA_adjustment")
                self._emit_vba_cmd({"trigger_mode": "Video"})
                self._emit_vba_cmd({"enabled": False})
                self._emit_fluo_cmd({"white": 512})
                for connected_module in self.connections.get("Folder VBA", []):
                    try:
                        connected_module.input_cb(
                            data={"record": False, "save": False}, data_type=IOTypes.CMD_DICT
                        )
                    except Exception:
                        pass

    def _on_skip_vba(self, *args: Any, **kwargs: Any) -> None:
        from core.app_state import app_state
        if app_state.mode != "dev":
            import os
            if not self.working_folder or not os.path.exists(self.working_folder) or not os.path.isdir(self.working_folder):
                from core.tutorial_manager import tutorial_manager
                tutorial_manager.play_error_hint(
                    self.working_folder_btn_tag, 
                    "No valid folder selected!\nPlease click 'Select Working Folder' to choose an existing destination directory."
                )
                logger.warning("Attempted to skip to VBA without selecting a valid folder")
                return
            
        self._set_step(7)
        self._emit_vba_cmd({"trigger_mode": "Video"})
        self._emit_vba_cmd({"enabled": False})
        self._emit_fluo_cmd({"white": 512})
        from core.module_registry import apply_named_view
        apply_named_view("VBA")

    # ═══════════════════════════════════════════════════════════════════════════
    # PIPELINE MODE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════

    def _set_mode(self, mode: str) -> None:
        """Switch to a pipeline mode and advance to its first step."""
        self.pipeline_mode = mode
        if mode == "capture":
            logger.info("TARA: Entering Data Capture pipeline")
            self._set_step(1)
        elif mode == "processing":
            logger.info("TARA: Entering Data Processing pipeline")
            # Reset processing-specific state for a clean run
            self._metadata_set_clicked = False
            self._csv_exported = False
            if dpg.does_item_exist(getattr(self, "proc_working_folder_label_tag", "")):
                dpg.set_value(self.proc_working_folder_label_tag, "None")
            self._set_step(1)
        elif mode == "processing_vba":
            logger.info("TARA: Entering Data Processing VBA pipeline")
            self._metadata_set_clicked = False
            self._picture_taken = False
            if dpg.does_item_exist(getattr(self, "proc_working_folder_label_tag", "")):
                dpg.set_value(self.proc_working_folder_label_tag, "None")
            self._set_step(1)

    # ── Capture pipeline ──────────────────────────────────────────────────────

    def _on_capture_next(self, step: int) -> None:
        """Advance the Capture pipeline by one step."""
        if step == 1:
            self._on_next_step0()          # folder validation → step 2
        elif step == 2:
            self._on_next_step1()          # exposure validation → step 3
        elif step == 3:
            self._on_next_step2()          # → step 4 (sequence view)
        elif step == 4:
            self._on_next_step3()          # protocol validation → step 5 (run sequence)
        elif step == 5:
            self._on_capture_next_step5()  # run sequence done → step 6 (position sample)
        elif step == 6:
            self._on_capture_next_step6()  # position sample → step 7 (VBA analysis)
        elif step == 7:
            self._on_capture_next_step7()  # VBA analysis done → step 8 (Done screen)

    def _on_capture_next_step7(self) -> None:
        """Capture step 7 → 8: validate image name & picture taken, then Done screen."""
        from core.app_state import app_state
        if app_state.mode != "dev":
            img_name = dpg.get_value(self.image_name_input_tag) if dpg.does_item_exist(self.image_name_input_tag) else ""
            if not img_name.strip():
                from core.tutorial_manager import tutorial_manager
                tutorial_manager.play_error_hint(
                    self.image_name_input_tag,
                    "Image name is missing!\nPlease enter a name for the image before proceeding."
                )
                return
            if not getattr(self, "_picture_taken", False):
                from core.tutorial_manager import tutorial_manager
                tutorial_manager.play_error_hint(
                    self.take_picture_btn_tag,
                    "Picture not taken!\nPlease click 'Take Picture' before proceeding."
                )
                return

        self._set_step(8)
        self._emit_vba_cmd({"enabled": False})
        self._emit_fluo_cmd({"white": 0})

    def _on_capture_next_step5(self) -> None:
        """Capture step 5 → 6: transition to Position Sample (VBA)."""
        from core.app_state import app_state
        if app_state.mode != "dev":
            if not getattr(self, "_sequence_started", False):
                from core.tutorial_manager import tutorial_manager
                tutorial_manager.play_error_hint(
                    self.start_sequence_btn_tag,
                    "Sequence not started!\nPlease click 'Start sequence' before proceeding."
                )
                return

            if getattr(self, "_sequence_running", False) or not getattr(self, "_sequence_stopped", False):
                from core.tutorial_manager import tutorial_manager
                tutorial_manager.play_error_hint(
                    self.start_sequence_btn_tag,
                    "Sequence is still running!\nPlease wait for the sequence to finish (STOP trigger) before proceeding."
                )
                logger.warning("Attempted to proceed from Step 5 before receiving sequence STOP trigger")
                return

        self._set_step(6)
        self._emit_vba_cmd({"trigger_mode": "Video"})
        self._emit_vba_cmd({"enabled": False})
        self._emit_fluo_cmd({"white": 512})
        from core.module_registry import apply_named_view
        apply_named_view("VBA_adjustment")

    def _on_capture_next_step6(self) -> None:
        """Capture step 6 → 7: activate VBA trigger for photo acquisition."""
        self._set_step(7)
        self._emit_vba_cmd({"trigger_mode": "Trigger"})
        self._emit_vba_cmd({"enabled": True})
        group_name = getattr(self, "last_group_name", "")
        if dpg.does_item_exist(getattr(self, "image_name_input_tag", "")):
            dpg.set_value(self.image_name_input_tag, f"{group_name}")
        from core.module_registry import apply_named_view
        apply_named_view("VBA")

    # ── Processing pipeline ───────────────────────────────────────────────────

    def _on_proc_next(self, step: int) -> None:
        """Advance the Processing pipeline by one step."""
        if step == 1:
            self._on_proc_next_step1()   # folder → metadata
        # step 2 = Metadata is terminal for processing

    def _on_proc_vba_next(self, step: int) -> None:
        """Advance the Processing VBA pipeline by one step."""
        if step == 1:
            self._on_proc_next_step1()   # folder → metadata (reused from Fluo)
        elif step == 2:
            self._on_proc_vba_next_step2()  # metadata → VBA Take Picture
        # step 3 = VBA Take Picture is terminal

    def _on_proc_vba_next_step2(self) -> None:
        """Processing VBA step 2 → 3: validate metadata, setup VBA trigger & view."""
        from core.app_state import app_state
        if app_state.mode != "dev":
            if not getattr(self, "_metadata_set_clicked", False):
                from core.tutorial_manager import tutorial_manager
                tutorial_manager.play_error_hint(
                    self.save_metadata_btn_tag,
                    "Metadata not saved!\n"
                    "Please click 'Save Metadata' to confirm before proceeding."
                )
                return

        self._picture_taken = False
        self._set_step(3)

        # Set white lights and cameras for photo acquisition
        self._emit_vba_cmd({"trigger_mode": "Trigger"})
        self._emit_vba_cmd({"enabled": True})
        self._emit_fluo_cmd({"white": 512})

        # Inject default group name into the image name input if available
        group_name = getattr(self, "last_group_name", "")
        if not group_name:
            # Reconstruct group name from metadata naming fields if possible
            fields = []
            for f in self.naming_fields_keys:
                val = self.naming_values.get(f, "")
                if val:
                    fields.append(str(val).strip().replace("-", "_"))
            if fields:
                group_name = "-".join(fields)
        if dpg.does_item_exist(getattr(self, "image_name_input_tag", "")):
            dpg.set_value(self.image_name_input_tag, f"{group_name}")

        from core.module_registry import apply_named_view
        apply_named_view("VBA")

    def _on_proc_next_step1(self) -> None:
        """Processing step 1 → 2: validate data folder, emit to image container."""
        from core.app_state import app_state
        if app_state.mode != "dev":
            if not self.working_folder or not os.path.exists(self.working_folder) or \
                    not os.path.isdir(self.working_folder):
                from core.tutorial_manager import tutorial_manager
                tutorial_manager.play_error_hint(
                    self.proc_working_folder_btn_tag,
                    "No valid data folder selected!\n"
                    "Please select the folder containing the raw images to process."
                )
                logger.warning("Attempted to proceed without a valid data folder")
                return

        # Emit the selected folder to appropriate modules depending on the mode
        if self.pipeline_mode == "processing_vba":
            self._emit_vba_cmd({"trigger_mode": "Trigger"})
            self._emit_vba_cmd({"enabled": True})
            for connected_module in self.connections.get("Folder VBA", []):
                try:
                    connected_module.input_cb(data=self.working_folder, data_type=IOTypes.FOLDER_PATH)
                except Exception as exc:
                    logger.error(f"Error sending VBA folder to {connected_module}: {exc}")
        else:
            # Emit the selected folder to the Fluo image container
            for connected_module in self.connections.get("Folder Fluo", []):
                try:
                    connected_module.input_cb(data=self.working_folder, data_type=IOTypes.FOLDER_PATH)
                except Exception as exc:
                    logger.error(f"Error sending folder to {connected_module}: {exc}")

        self._metadata_set_clicked = False
        self._set_step(2)
        from core.module_registry import apply_named_view
        if self.pipeline_mode == "processing_vba":
            apply_named_view("VBA_post_process")
        else:
            apply_named_view("sample_selection")

    # ── Processing backward navigation ────────────────────────────────────────

    def _on_proc_step1_prev(self) -> None:
        """Go back from processing step 1 to the mode-selection screen."""
        self.pipeline_mode = None
        self._set_step(0)

    def _on_proc_prev_from_ui(self, ui_step: int) -> None:
        """Handle Prev pressed inside the reused step6 group header (processing mode).

        Args:
            ui_step: The original step number embedded in the group header
                     (6 = Metadata).
        """
        # Map the UI step number back to the protocol_step used in processing mode
        ui_to_proto = {6: 2}
        proto_step = ui_to_proto.get(ui_step, ui_step)
        target_proto_step = proto_step - 1

        # Clean up state we are leaving
        if proto_step == 2:     # leaving Metadata (step6)
            self._metadata_set_clicked = False
            for f in self.naming_fields_keys:
                field_config = self.field_config.get(f, {})
                if not field_config.get("keep", True):
                    options = field_config.get("options", None)
                    if options:
                        combo_tag = self.naming_fields_combo_tags[f]
                        custom_tag = self.naming_fields_custom_tags[f]
                        default_val = field_config.get("default", "") or ""
                        if dpg.does_item_exist(combo_tag):
                            if default_val in options:
                                dpg.set_value(combo_tag, default_val)
                                if dpg.does_item_exist(custom_tag):
                                    dpg.configure_item(custom_tag, show=False)
                            else:
                                dpg.set_value(combo_tag, "Custom")
                                if dpg.does_item_exist(custom_tag):
                                    dpg.set_value(custom_tag, default_val)
                                    dpg.configure_item(custom_tag, show=True)
                        self.naming_values[f] = default_val
                    else:
                        if dpg.does_item_exist(self.naming_fields_tags[f]):
                            dpg.set_value(self.naming_fields_tags[f], "")
                        self.naming_values[f] = ""

        self._set_step(target_proto_step)

        # Restore appropriate view / hardware state
        from core.module_registry import apply_named_view
        if target_proto_step == 2:
            apply_named_view("sample_selection")
            self._emit_fluo_cmd({"white": 0})

    def _on_proc_vba_prev_from_ui(self, ui_step: int) -> None:
        """Handle Prev pressed in processing_vba mode.
        ui_step can be:
          - 6: Metadata -> Step 1 (Folder Select)
        """
        if ui_step == 6:  # From Step 2 (Metadata) to Step 1 (Folder Select)
            self.working_folder = None
            self.current_folder = None
            if dpg.does_item_exist(self.proc_working_folder_label_tag):
                dpg.set_value(self.proc_working_folder_label_tag, "None")
            self._set_step(1)
            from core.module_registry import apply_named_view
            apply_named_view("new_exp")

    # ── Processing folder picker ──────────────────────────────────────────────

    def _on_proc_working_folder_clicked(self, *args: Any, **kwargs: Any) -> None:
        """Open folder picker for the processing data folder."""
        from core.file_explorer import file_explorer
        default_path = self.working_folder or self.master_folder or ""
        file_explorer.select_folder(
            default_path=default_path,
            callback=self._on_proc_working_folder_selected
        )

    def _on_proc_working_folder_selected(self, selected_dir: str | None, *args: Any, **kwargs: Any) -> None:
        """Called when the user picks a data folder in processing mode."""
        if selected_dir:
            self.working_folder = selected_dir
            self.current_folder = selected_dir
            if dpg.does_item_exist(self.proc_working_folder_label_tag):
                dpg.set_value(
                    self.proc_working_folder_label_tag,
                    f".../{os.path.basename(selected_dir)}"
                )
            self._auto_detect_date_from_path(selected_dir)
            logger.info(f"Processing data folder set to: {selected_dir}")
            self._update_step_highlights()

    def _auto_detect_date_from_path(self, path: str) -> None:
        if not path:
            return
        
        path = path.replace("\\", "/")
        parts = path.split("/")
        
        detected_date = None
        import re
        date_pattern = re.compile(r'^(19|20|21)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$')
        
        for part in reversed(parts):
            if date_pattern.match(part):
                detected_date = part
                break
                
        if detected_date:
            logger.info(f"Auto-detected date folder '{detected_date}' in path '{path}'")
            if "Date" in self.naming_fields_keys:
                self.naming_values["Date"] = detected_date
                tag = self.naming_fields_tags.get("Date")
                if tag and dpg.does_item_exist(tag):
                    dpg.set_value(tag, detected_date)
        else:
            today_date = datetime.date.today().strftime("%Y%m%d")
            logger.debug(f"No date folder (YYYYMMDD) found in path '{path}', falling back to '{today_date}'")
            if "Date" in self.naming_fields_keys:
                self.naming_values["Date"] = today_date
                tag = self.naming_fields_tags.get("Date")
                if tag and dpg.does_item_exist(tag):
                    dpg.set_value(tag, today_date)


    def _on_master_folder_clicked(self, *args: Any, **kwargs: Any) -> None:
        from core.file_explorer import file_explorer
        file_explorer.select_folder(default_path=self.master_folder or "", callback=self._on_master_folder_selected)

    def _on_master_folder_selected(self, selected_dir: str | None, *args: Any, **kwargs: Any) -> None:
        if selected_dir:
            self.master_folder = selected_dir
            self.last_date_folder = None
            if dpg.does_item_exist(self.master_folder_label_tag):
                dpg.set_value(self.master_folder_label_tag, f".../{os.path.basename(selected_dir)}")
            logger.info(f"Master folder set to: {selected_dir}")

    def _on_working_folder_clicked(self, *args: Any, **kwargs: Any) -> None:
        from core.file_explorer import file_explorer
        # Default to the last date folder if it exists, otherwise current working folder
        default_path = self.working_folder
        if not default_path and self.last_date_folder:
            default_path = self.last_date_folder
        elif not default_path and self.master_folder and self._subfolders_created:
            date_val = dpg.get_value(self.main_subfolder_tag).strip() if dpg.does_item_exist(self.main_subfolder_tag) else datetime.date.today().strftime("%Y%m%d")
            default_path = os.path.join(self.master_folder, date_val)
        elif not default_path and self.master_folder:
            default_path = self.master_folder
        file_explorer.select_folder(default_path=default_path or "", callback=self._on_working_folder_selected)

    def _on_working_folder_selected(self, selected_dir: str | None, *args: Any, **kwargs: Any) -> None:
        if selected_dir:
            self.working_folder = selected_dir
            self.current_folder = selected_dir
            if dpg.does_item_exist(self.working_folder_label_tag):
                dpg.set_value(self.working_folder_label_tag, f".../{os.path.basename(selected_dir)}")

            self._auto_detect_date_from_path(selected_dir)

            for k in ["Folder Fluo", "Folder VBA"]:
                for connected_module in self.connections.get(k, []):
                    try:
                        target_dir = selected_dir
                        if k == "Folder VBA":
                            target_dir = os.path.join(selected_dir, "VBA")
                            if not os.path.exists(target_dir):
                                os.makedirs(target_dir, exist_ok=True)
                        connected_module.input_cb(data=target_dir, data_type=IOTypes.FOLDER_PATH)
                    except Exception as e:
                        logger.error(f"Error sending folder to {connected_module}: {e}")
            # Also ping protocol file just in case
            for connected_module in self.connections.get("Protocol File", []):
                try:
                    connected_module.input_cb(data=selected_dir, data_type=IOTypes.FOLDER_PATH)
                except Exception: pass
                
            logger.info(f"Working folder set to: {selected_dir}")
            self._update_step_highlights()

    def _on_create_subfolders_clicked(self, *args: Any, **kwargs: Any) -> None:
        import os
        from core.tutorial_manager import tutorial_manager
        
        if not self.master_folder or not os.path.exists(self.master_folder) or not os.path.isdir(self.master_folder):
            tutorial_manager.play_error_hint(
                self.master_folder_btn_tag, 
                "Please select a valid master folder first!"
            )
            logger.warning("Attempted to create subfolders without a valid master folder selected")
            return
            
        # Get date from input text
        main_val = dpg.get_value(self.main_subfolder_tag).strip() if dpg.does_item_exist(self.main_subfolder_tag) else datetime.date.today().strftime("%Y%m%d")
        fvfm_val = "FvFm"
        superseq_val = "SuperSeq"
        
        if not main_val:
            tutorial_manager.play_error_hint(
                self.create_subfolders_btn_tag,
                "Date must be provided!"
            )
            logger.warning("Attempted to create subfolders with empty date")
            return
            
        # Clean folder names
        from core.tools import TextTools, Charsets
        for name, val in [("Date Folder", main_val), ("FvFm Subfolder", fvfm_val), ("SuperSeq Subfolder", superseq_val)]:
            is_valid, invalid_chars = TextTools.check(text=val, allowed_chars=Charsets.ALPHANUMERIC_SPACE + "-_")
            if not is_valid:
                chars_display = "', '".join(invalid_chars)
                tutorial_manager.play_error_hint(
                    self.create_subfolders_btn_tag,
                    f"Invalid characters found in {name}: '{chars_display}'"
                )
                logger.warning(f"Invalid characters in folder name {name}: {invalid_chars}")
                return

        # Build paths
        main_dir = os.path.join(self.master_folder, main_val)
        fvfm_dir = os.path.join(main_dir, fvfm_val)
        superseq_dir = os.path.join(main_dir, superseq_val)
        
        # Create directories if they don't exist
        try:
            created_any = False
            for d in [main_dir, fvfm_dir, superseq_dir]:
                if not os.path.exists(d):
                    os.makedirs(d, exist_ok=True)
                    created_any = True
            
            # Set working folder to the date folder by default
            self.working_folder = main_dir
            self.current_folder = main_dir
            self.last_date_folder = main_dir  # Remember this for future resets
            if dpg.does_item_exist(self.working_folder_label_tag):
                dpg.set_value(self.working_folder_label_tag, f".../{os.path.basename(main_dir)}")
            
            self._subfolders_created = True
            self._update_step_highlights()
            
            msg = "Date folder structure created successfully!" if created_any else "Date folder structure already exists."
            tutorial_manager.play_success_hint(
                self.create_subfolders_btn_tag,
                msg,
                icon_type="success"
            )
            logger.info(f"Date folder structure configured successfully: {main_dir} with FvFm and SuperSeq subfolders")
            
        except Exception as e:
            logger.error(f"Failed to create subfolders: {e}")
            tutorial_manager.play_error_hint(
                self.create_subfolders_btn_tag,
                f"Error creating subfolders: {e}"
            )

    def _on_next_step0(self) -> None:
        from core.app_state import app_state
        if app_state.mode != "dev":
            import os
            if not self.working_folder or not os.path.exists(self.working_folder) or not os.path.isdir(self.working_folder):
                from core.tutorial_manager import tutorial_manager
                tutorial_manager.play_error_hint(
                    self.working_folder_btn_tag, 
                    "No valid working folder selected!\nPlease click 'Select Working Folder' to choose an existing destination directory."
                )
                logger.warning("Attempted to proceed without selecting a valid working folder")
                return
        self._set_step(2)
        self._emit_fluo_cmd({"trigger_mode": "Trigger"})
        from core.module_registry import apply_named_view
        apply_named_view("calibration")
        
    def _on_next_step1(self) -> None:
        from core.app_state import app_state
        if app_state.mode != "dev":
            exp_val = dpg.get_value(self.best_exp_int_tag)
            if exp_val <= 0:
                from core.tutorial_manager import tutorial_manager
                tutorial_manager.play_error_hint(
                    self.best_exp_int_tag, 
                    "Exposure not configured!\nPlease set a valid exposure or use 'Find best expos'."
                )
                logger.warning("Attempted to proceed without valid exposure")
                return
            
        self._set_step(3)
        self._emit_fluo_cmd({"clear": True})

    def _on_next_step2(self) -> None:
        self._set_step(4)
        
        from core.module_registry import apply_named_view
        apply_named_view("sequence")

    def _on_next_step3(self) -> None:
        from core.app_state import app_state
        if app_state.mode != "dev":
            if getattr(self, "protocol_combo_tag", None) and not dpg.get_value(self.protocol_combo_tag):
                from core.tutorial_manager import tutorial_manager
                tutorial_manager.play_error_hint(
                    self.protocol_combo_tag, 
                    "No protocol selected!\nPlease select a protocol from the dropdown."
                )
                return

            if getattr(self, "group_name_input_tag", None) and not dpg.get_value(self.group_name_input_tag).strip():
                from core.tutorial_manager import tutorial_manager
                tutorial_manager.play_error_hint(
                    self.group_name_input_tag, 
                    "Sample / Group name is missing!\nPlease provide a name."
                )
                return

            if not getattr(self, "_protocol_set_clicked", False):
                from core.tutorial_manager import tutorial_manager
                tutorial_manager.play_error_hint(
                    self.run_protocol_btn_tag, 
                    "Protocol not applied!\nPlease click 'Set Protocol' before proceeding."
                )
                return
            
        self._set_step(5)


    def _on_protocol_input_changed(self, sender: Any, app_data: Any, *args: Any, **kwargs: Any) -> None:
        self._protocol_set_clicked = False
        self._update_step_highlights()

    def _on_combo_changed(self, sender: Any, app_data: Any, user_data: Any, *args: Any, **kwargs: Any) -> None:
        """Handle dropdown selection change."""
        field = user_data
        custom_tag = self.naming_fields_custom_tags[field]
        
        if app_data == "Custom":
            # Show custom input field
            if dpg.does_item_exist(custom_tag):
                dpg.configure_item(custom_tag, show=True)
        else:
            # Hide custom input field and update value
            if dpg.does_item_exist(custom_tag):
                dpg.configure_item(custom_tag, show=False)
            self.naming_values[field] = app_data
        self._metadata_set_clicked = False
        self._emit_metadata()

    def _update_naming_field(self, sender: Any, app_data: Any, user_data: Any, *args: Any, **kwargs: Any) -> None:
        self.naming_values[user_data] = app_data
        self._metadata_set_clicked = False
        self._emit_metadata()

    def _update_target_signal(self, sender: Any, app_data: Any, *args: Any, **kwargs: Any) -> None:
        self.target_signal = app_data

    # ── Progress popup helpers (image_container pattern) ─────────────────────

    def _show_progress(self, expected_frames: int) -> None:
        """Show the (always-modal) progress popup and re-centre it after DPG renders one frame."""
        dpg.set_value(self._progress_bar_tag, 0.0)
        dpg.set_value(self._progress_text_tag, f"Acquisition…  (0 / {expected_frames})")

        # Show first so DPG can measure the actual autosize dimensions
        dpg.configure_item(self._progress_popup_tag, show=True)

        # Re-centre after one frame so the real rect_size is available
        def _recentre() -> None:
            import time
            time.sleep(0.05)   # ~1 frame at 20 fps minimum
            if dpg.does_item_exist(self._progress_popup_tag):
                vw = dpg.get_viewport_width()
                vh = dpg.get_viewport_height()
                w, h = dpg.get_item_rect_size(self._progress_popup_tag)
                if w > 0 and h > 0:
                    dpg.set_item_pos(self._progress_popup_tag,
                                     [max(0, (vw - w) // 2), max(0, (vh - h) // 2)])

        import threading
        threading.Thread(target=_recentre, daemon=True).start()
        logger.debug(f"TARA progress popup shown (expected={expected_frames})")

    def _update_progress(self, received: int, total: int) -> None:
        """Update progress bar and label text."""
        if not dpg.does_item_exist(self._progress_bar_tag):
            return
        fraction = (received / total) if total > 0 else 0.0
        dpg.set_value(self._progress_bar_tag, fraction)
        dpg.set_value(self._progress_text_tag, f"Acquisition…  ({received} / {total})")

    def _hide_progress(self) -> None:
        """Hide the progress popup and reset find-exp state."""
        self._find_exp_active = False
        if dpg.does_item_exist(self._progress_popup_tag):
            dpg.configure_item(self._progress_popup_tag, show=False)
        logger.debug("TARA progress popup hidden")

    def _on_progress_skip(self, *args: Any, **kwargs: Any) -> None:
        """Advanced/dev only: manually count +1 frame to help debug the popup."""
        if not self._find_exp_active:
            return
        self._find_exp_frame_count += 1
        received = self._find_exp_frame_count
        total    = self._find_exp_expected_frames
        self._update_progress(received, total)
        if received >= total:
            self._hide_progress()

    def _on_key_delete(self, sender: Any, app_data: Any, user_data: Any = None) -> None:
        """Allow manual frame skip in User Mode using DEL key when progress bar is active."""
        if self._find_exp_active:
            logger.info("Delete key pressed while waiting for frames. Manually injecting 1 frame.")
            self._on_progress_skip()

    # ── Find best exposure command ────────────────────────────────────────────

    def _send_find_exp_cmd(self, *args: Any, **kwargs: Any) -> None:
        # Number of frames the IDS camera will produce for fluo_fixed_int_test.
        # Keep this in sync with whatever the camera / Fluo module actually captures.
        expected = 50

        # Reset counter and activate tracking
        self._find_exp_frame_count = 0
        self._find_exp_expected_frames = expected
        self._find_exp_active = True

        self._show_progress(expected)

        cmd = {"fluo_fixed_int_test": True}
        logger.debug(f"{self.winID} sending find optimal exposure cmd")
        self._emit_fluo_cmd(cmd)

    def _send_set_expos_cmd(self, *args: Any, **kwargs: Any) -> None:
        if dpg.does_item_exist(self.best_exp_int_tag):
            val = dpg.get_value(self.best_exp_int_tag)
            if val > 0:
                cmd = {"exposure": val}
                logger.debug(f"{self.winID} sending set exposure cmd={cmd}")
                self._emit_fluo_cmd(cmd)
                self._update_step_highlights()

                from core.app_state import app_state
                app_state.detector = val
            else:
                logger.warning(f"Invalid exposure value (must be > 0), got: {val}")

    def _send_test_detection_cmd(self, *args: Any, **kwargs: Any) -> None:
        exp_val = dpg.get_value(self.best_exp_int_tag)
        if exp_val <= 0:
            logger.warning("Attempted test detection without setting exposure.")
            return

        w, h = 1280, 1024
        black_frame = np.zeros((h, w), dtype=np.uint8)
        data = (black_frame, "TARA")
        for module in self.connections.get("Frame", []):
            try:
                module.input_cb(data)
                logger.info(f"Emitted black frame to {module.label} to clear downstream visualization")
            except Exception as e:
                logger.error(f"Failed to emit black frame to {module}: {e}")

        for connected_module in self.connections.get("Folder Fluo", []):
            try:
                connected_module.input_cb(data={"clear": True}, data_type=IOTypes.CMD_DICT)
            except Exception:
                pass

        # Reset counter and activate tracking
        expected = 1
        self._find_exp_frame_count = 0
        self._find_exp_expected_frames = expected
        self._find_exp_active = True

        self._show_progress(expected)

        def _task() -> None:
            logger.info("Sending reference frame trigger...")
            cmd_ref = {"fluo_ref": True, "n_frames": expected, "exposure": exp_val}
            self._emit_fluo_cmd(cmd_ref)
            
            import time
            wait_time = expected * (exp_val / 1000000.0) + 2.0
            time.sleep(wait_time)
            
            logger.info("Sending test detection trigger...")
            cmd_test = {"testdet": True}
            self._emit_fluo_cmd(cmd_test)

        import threading
        threading.Thread(target=_task, daemon=True).start()
        
        self._test_detection_clicked = True
        self._update_step_highlights()

    def _send_protocol_cmd(self, *args: Any, **kwargs: Any) -> None:
        selected = dpg.get_value(self.protocol_combo_tag)
        group_name = dpg.get_value(self.group_name_input_tag).strip()

        if not selected:
            from core.tutorial_manager import tutorial_manager
            tutorial_manager.play_error_hint(
                self.protocol_combo_tag,
                "No protocol selected!\nPlease select a protocol from the dropdown."
            )
            logger.warning("No protocol selected")
            return
            
        if not group_name:
            from core.tutorial_manager import tutorial_manager
            tutorial_manager.play_error_hint(
                self.group_name_input_tag,
                "Sample / Group name is missing!\nPlease provide a name before running the protocol."
            )
            logger.warning("No group name provided")
            return

        from core.tools import TextTools, Charsets
        is_valid, invalid_chars = TextTools.check(
            text=group_name, 
            allowed_chars=Charsets.ALPHANUMERIC_SPACE
        )
        if not is_valid:
            from core.tutorial_manager import tutorial_manager
            chars_display = "', '".join(invalid_chars)
            tutorial_manager.play_error_hint(
                self.group_name_input_tag,
                f"Invalid characters found: '{chars_display}'\nPlease use only letters, numbers, and spaces."
            )
            logger.warning(f"Invalid characters in group name: {invalid_chars}")
            return

        self.last_group_name = group_name
        
        # Get best exposure time from step 2
        best_exp = dpg.get_value(self.best_exp_int_tag) if dpg.does_item_exist(self.best_exp_int_tag) else 0
            
        import os
        base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "sequence_processor", "sequences")
        file_path = os.path.join(base_dir, selected)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Inject variable definitions at the beginning
            content = f"var det_time={best_exp},\n" + content
            content = f"G{group_name},\n" + content
                
            for connected_module in self.connections.get("Protocol File", []):
                try:
                    connected_module.input_cb(data=content, data_type=IOTypes.TEXT)
                    connected_module.input_cb(data={"validate": True}, data_type=IOTypes.CMD_DICT)
                except Exception as e:
                    logger.error(f"Error sending protocol to {connected_module}: {e}")
            
            self._protocol_set_clicked = True
            logger.success(f"{self.label} sent protocol '{selected}' with group '{group_name}'")
            self._update_step_highlights()

        except Exception as e:
            logger.error(f"Failed to read protocol '{selected}': {e}")

    def _send_start_sequence_cmd(self, *args: Any, **kwargs: Any) -> None:
        cmd = {"start_sequence": True}
        logger.debug(f"{self.winID} sending start sequence cmd")
        self._emit_fluo_cmd(cmd)
        self._sequence_started = True
        self._sequence_running = True
        self._sequence_stopped = False
        self._update_step_highlights()

    def _send_take_picture_cmd(self, *args: Any, **kwargs: Any) -> None:
        img_name = dpg.get_value(self.image_name_input_tag)
        cmd = {
            "record": True,
            "save": True,
            "names": [img_name]
        }
        
        for connected_module in self.connections.get("Folder VBA", []):
            try:
                connected_module.input_cb(data=cmd, data_type=IOTypes.CMD_DICT)
            except Exception:
                pass
                
        self._emit_vba_cmd({"trigger_cam": True})
        self._picture_taken = True
        self._update_step_highlights()

    def _emit_metadata(self) -> None:
        fields = []
        metadata_dict = {}
        for f in self.naming_fields_keys:
            field_config = self.field_config.get(f, {})
            options = field_config.get("options", None)
            show_ui = field_config.get("show_ui", True)
            
            if show_ui:
                if options:
                    combo_tag = self.naming_fields_combo_tags[f]
                    custom_tag = self.naming_fields_custom_tags[f]
                    combo_val = dpg.get_value(combo_tag) if dpg.does_item_exist(combo_tag) else ""
                    if combo_val == "Custom":
                        val = dpg.get_value(custom_tag) if dpg.does_item_exist(custom_tag) else ""
                    else:
                        val = combo_val
                else:
                    val = dpg.get_value(self.naming_fields_tags[f])
            else:
                val = self.naming_values.get(f, "")
                if val is None or str(val).strip() == "":
                    val = field_config.get("default")
            
            val_str = str(val).strip() if val is not None else ""
            if val_str:
                fields.append(val_str.replace("-", "_"))
            metadata_dict[f] = val_str
        
        group_name_base = getattr(self, "last_group_name", "")
        if group_name_base:
            fields.append(group_name_base.replace("-", "_"))
        fields.append("PAM")
        group_name = "-".join(fields)

        editability = {}
        populate_default_key = "None"
        fill_with_data_key = "PAM_value"
        optional_fields = []
        for f in self.naming_fields_keys:
            field_config = self.field_config.get(f, {})
            editability[f] = field_config.get("editable", True)
            if field_config.get("populate_default", False):
                populate_default_key = f
            if field_config.get("fill_with_data", False):
                fill_with_data_key = f
            if field_config.get("default") is None:
                optional_fields.append(f)

        cmd = {
            "metadata": metadata_dict, 
            "full_name": group_name, 
            "editable": editability,
            "populate_default": populate_default_key,
            "fill_with_data_key": fill_with_data_key,
            "optional_fields": optional_fields
        }

        for connected_module in self.connections.get("Metadata", []):
            try:
                connected_module.input_cb(data=cmd, data_type=IOTypes.CMD_DICT)
            except Exception as e:
                logger.error(f"Error sending metadata to {connected_module}: {e}")

    def _send_metadata_cmd(self, *args: Any, **kwargs: Any) -> None:
        for f in self.naming_fields_keys:
            field_config = self.field_config.get(f, {})
            options = field_config.get("options", None)
            show_ui = field_config.get("show_ui", True)
            
            if show_ui:
                if options:
                    combo_tag = self.naming_fields_combo_tags[f]
                    custom_tag = self.naming_fields_custom_tags[f]
                    combo_val = dpg.get_value(combo_tag) if dpg.does_item_exist(combo_tag) else ""
                    
                    if combo_val == "Custom":
                        val = dpg.get_value(custom_tag) if dpg.does_item_exist(custom_tag) else ""
                    else:
                        val = combo_val
                else:
                    val = dpg.get_value(self.naming_fields_tags[f])
            else:
                val = self.naming_values.get(f, "")
                if val is None or str(val).strip() == "":
                    val = field_config.get("default")
            
            default_val = field_config.get("default")
            
            if val is None or str(val).strip() == "":
                if default_val is None or not show_ui:
                    pass
                else:
                    from core.tutorial_manager import tutorial_manager
                    error_tag = self.naming_fields_custom_tags[f] if options and dpg.does_item_exist(self.naming_fields_combo_tags[f]) and dpg.get_value(self.naming_fields_combo_tags[f]) == "Custom" else (self.naming_fields_combo_tags[f] if options else self.naming_fields_tags[f])
                    tutorial_manager.play_error_hint(
                        error_tag,
                        f"Parameter {f} must be completed"
                    )
                    logger.warning(f"Missing mandatory field: {f}")
                    return

        # Emit the gathered metadata downstream
        self._emit_metadata()
        
        # Clear fields based on keep flag from config
        for f in self.naming_fields_keys:
            field_config = self.field_config.get(f, {})
            should_keep = field_config.get("keep", True)
            show_ui = field_config.get("show_ui", True)
            
            if not should_keep:
                if show_ui:
                    options = field_config.get("options", None)
                    if options:
                        # Handle dropdown
                        combo_tag = self.naming_fields_combo_tags[f]
                        custom_tag = self.naming_fields_custom_tags[f]
                        
                        # Reset to default
                        default_val = field_config.get("default", "")
                        if default_val is None:
                            default_val = ""
                        
                        if dpg.does_item_exist(combo_tag):
                            if default_val in options:
                                dpg.set_value(combo_tag, default_val)
                                if dpg.does_item_exist(custom_tag):
                                    dpg.configure_item(custom_tag, show=False)
                            else:
                                dpg.set_value(combo_tag, "Custom")
                                if dpg.does_item_exist(custom_tag):
                                    dpg.set_value(custom_tag, default_val)
                                    dpg.configure_item(custom_tag, show=True)
                        
                        self.naming_values[f] = default_val
                    else:
                        # Handle regular text input
                        if dpg.does_item_exist(self.naming_fields_tags[f]):
                            dpg.set_value(self.naming_fields_tags[f], "")
                        self.naming_values[f] = ""
                else:
                    self.naming_values[f] = field_config.get("default", "") or ""

        self._metadata_set_clicked = True
        self._update_step_highlights()

    def _emit_fluo_cmd(self, cmd: dict[str, Any]) -> None:
        for connected_module in self.connections.get("Fluo CMD", []):
            try:
                connected_module.input_cb(data=cmd, data_type=IOTypes.CMD_DICT)
            except Exception as e:
                logger.error(f"Error sending fluo cmd to {connected_module}: {e}")

    def _emit_vba_cmd(self, cmd: dict[str, Any]) -> None:
        for connected_module in self.connections.get("VBA CMD", []):
            try:
                connected_module.input_cb(data=cmd, data_type=IOTypes.CMD_DICT)
            except Exception as e:
                logger.error(f"Error sending vba cmd to {connected_module}: {e}")

    def input_cb(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Handle incoming data (SAMPLE, FOLDER_PATH, FRAME12, or TRIGGER)."""
        import os

        data_val = kwargs.get("sample", kwargs.get("data", data))
        data_type = kwargs.get("data_type", None)

        if data_type == IOTypes.TRIGGER or (isinstance(data_val, str) and data_val in ("start", "stop")):
            if data_val == "start":
                self._sequence_running = True
                self._sequence_stopped = False
                logger.info(f"{self.label} received sequence START trigger")
            elif data_val == "stop":
                self._sequence_running = False
                self._sequence_stopped = True
                logger.info(f"{self.label} received sequence STOP trigger")
            return

        _is_frame = (
            data_type in (IOTypes.FRAME12, IOTypes.FRAME)
            or (
                data_type is None
                and isinstance(data_val, tuple)
                and len(data_val) == 2
                and hasattr(data_val[0], "shape")
            )
        )
        if _is_frame:
            if self._find_exp_active:
                self._find_exp_frame_count += 1
                received = self._find_exp_frame_count
                total    = self._find_exp_expected_frames
                self._update_progress(received, total)
                if received >= total:
                    self._hide_progress()
            return  # Frames are not forwarded further by TARA

        if data_type == IOTypes.FOLDER_PATH or (isinstance(data_val, str) and os.path.isdir(data_val)):
            self.current_folder = data_val
            self._subfolders_created = False
            logger.info(f"{self.label} received folder path: {self.current_folder}")
            
            # Forward FOLDER_PATH to Protocol File so Sequence Processor knows where to save
            for connected_module in self.connections.get("Protocol File", []):
                try:
                    connected_module.input_cb(data=self.current_folder, data_type=IOTypes.FOLDER_PATH)
                except Exception:
                    pass
            return

        if data_type == IOTypes.SAMPLE or isinstance(data_val, dict):
            # Only auto-detect/set exposure if we are in Capture mode and at Step 2
            if self.pipeline_mode != "capture" or getattr(self, "protocol_step", 0) != 2:
                return

            # Try to process it as a SAMPLE format dict
            sample = data_val
            if "x" in sample and "y" in sample:
                try:
                    x_data = np.array(sample["x"])
                    y_data = np.array(sample["y"])
                    
                    if len(y_data) > 0 and len(x_data) == len(y_data):
                        # Find the index of y closest to target_signal
                        idx = np.abs(y_data - self.target_signal).argmin()
                        best_expos = int(x_data[idx])
                        
                        if dpg.does_item_exist(self.best_exp_int_tag):
                            dpg.set_value(self.best_exp_int_tag, best_expos)
                            self._send_set_expos_cmd()

                        # If the SAMPLE arrives while the progress popup is still
                        # open (e.g. frame count was off), close it now.
                        if self._find_exp_active:
                            self._hide_progress()
                            
                        logger.success(f"{self.label} found best exposure: {best_expos} for target signal {self.target_signal} (actual signal {y_data[idx]:.1f})")
                except Exception as e:
                    logger.error(f"Error processing sample in {self.label}: {e}")

EXPORTED_CLASS = TaraWin
EXPORTED_NAME = "TARA"
