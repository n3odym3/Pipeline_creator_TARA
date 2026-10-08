import re
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
import dearpygui.dearpygui as dpg

from modules.nodal_data_processor.data_processor_base import DataProcessorBase


class TaraProcessor(DataProcessorBase):
    """Automatically parses and splits sequences from TARA experiments (Raw, FvFm, Phi PSII, OJIP, rETR, NPQ)."""

    DEFAULT_BLOCKS: str = "126-137, 200-209, 272-281, 344-353, 416-425, 488-497"
    DEFAULT_ACTINIC: List[float] = [30.0, 60.0, 120.0, 250.0, 500.0, 750.0]
    DEFAULT_DARK_RELAX: str = "559-581"

    def __init__(self, node_id: Union[int, str]):
        super().__init__(node_id)
        
        # Define exporter parameter
        self.export_type: str = "Raw"
        self.tooltip_text: Union[int, str] = -1

        # rETR parameters
        self.retr_mode: str = "Average"         # "Average" or "Index"
        self.retr_index: int = -1              # -1 for last, 0 for 1st, 2 for 3rd, etc.
        self.retr_blocks: str = self.DEFAULT_BLOCKS
        self.retr_pair_f0_fm: bool = True
        self.retr_x_axis: str = "Actinic Intensity"  # "Actinic Intensity" or "Palier Index"
        self.retr_ui_group: Union[int, str] = -1
        self.retr_index_widget: Union[int, str] = -1

        # NPQ parameters
        self.npq_mode: str = "Last Lightstep"   # "Last Lightstep" or "All Data"
        self.npq_calc_type: str = "NPQ"        # "NPQ" or "qN"
        self.npq_use_max_fm: bool = False
        self.npq_initial_fm_index: int = 1     # Index of initial dark Fm
        self.npq_dark_relax: str = self.DEFAULT_DARK_RELAX
        self.npq_ui_group: Union[int, str] = -1
        self.npq_dark_relax_widget: Union[int, str] = -1
        
        # Override name for display
        self.name: str = "TARA Exporter"

    def parse_blocks(self, text: str) -> List[Tuple[int, int]]:
        """
        Parse block ranges from string format (e.g. '126-137, 200-209').
        Auto-corrects 416-525 to 416-425 if it would overlap subsequent block at 488.
        """
        matches = re.findall(r'(\d+)\s*[-:]\s*(\d+)', text)
        blocks: List[Tuple[int, int]] = []
        for start_str, end_str in matches:
            s, e = int(start_str), int(end_str)
            if s > e:
                s, e = e, s
            blocks.append((s, e))

        # Check for 416-525 typo overlapping with 488
        for i, (s, e) in enumerate(blocks):
            if s == 416 and e == 525:
                if i + 1 < len(blocks) and blocks[i + 1][0] < 525:
                    blocks[i] = (416, 425)

        return blocks

    def get_actinic_intensity(self, block_labels: List[str], block_idx: int, fallback_x: float) -> float:
        """
        Extract actinic intensity from Ax pattern in labels (e.g. A30, _A60_, A120).
        If not found in labels, falls back to standard TARA sequence actinic intensities:
        [30, 60, 120, 250, 500, 750], or fallback_x.
        """
        pattern = re.compile(r'(?:^|[_\-\s/\\\.])A(\d+(?:\.\d+)?)(?:[_\-\s/\\\.+]|$)', re.IGNORECASE)
        fallback_pattern = re.compile(r'A(\d+(?:\.\d+)?)', re.IGNORECASE)

        for label in block_labels:
            if not label:
                continue
            stem = Path(label.split('+')[0]).stem
            m = pattern.search(stem)
            if not m:
                m = fallback_pattern.search(stem)
            if m:
                try:
                    return float(m.group(1))
                except (ValueError, IndexError):
                    pass

        if block_idx < len(self.DEFAULT_ACTINIC):
            return self.DEFAULT_ACTINIC[block_idx]

        return fallback_x

    def _process_phi_psii(
        self,
        x_list: List[float],
        y_list: List[float],
        labels: Optional[List[str]] = None
    ) -> Tuple[List[float], List[float], Optional[List[str]]]:
        """
        Compute Phi PSII = (Fm - F0) / Fm for ALL F0/Fm pairs across the sequence.
        Matches FvFmProcessor logic: filters data using F0/Fm label patterns to skip OJIP and other frames,
        and pairs them by timestamp.
        """
        if labels and len(labels) == len(y_list):
            res = self._process_phi_psii_with_labels(x_list, y_list, labels)
            if res is not None:
                return res

        # Fallback: sequential pairing of whatever points are present, ignoring OJIP if labeled
        filtered_indices = []
        for i in range(len(y_list)):
            if labels and i < len(labels) and "OJIP" in labels[i].upper():
                continue
            filtered_indices.append(i)

        out_x: List[float] = []
        out_y: List[float] = []
        out_labels: List[str] = []
        for k in range(0, len(filtered_indices) - 1, 2):
            i0 = filtered_indices[k]
            im = filtered_indices[k + 1]
            f0 = y_list[i0]
            fm = y_list[im]
            phi = (fm - f0) / fm if fm != 0 else 0.0
            out_x.append(x_list[im])
            out_y.append(phi)
            if labels and im < len(labels):
                out_labels.append(f"{labels[i0]}+{labels[im]}")

        return out_x, out_y, out_labels if labels else None

    def _process_phi_psii_with_labels(
        self,
        x_list: List[float],
        y_list: List[float],
        labels: List[str]
    ) -> Optional[Tuple[List[float], List[float], List[str]]]:
        """Smart F0/Fm pairing matching FvFmProcessor (skips OJIP and non-F0/Fm points)."""
        f0_pattern = re.compile(r'(?:^|[_\-])F0(?:[_\-]|$)', re.IGNORECASE)
        fm_pattern = re.compile(r'(?:^|[_\-])Fm(?:[_\-]|$)', re.IGNORECASE)

        f0_items: List[Tuple[int, int, float, float, str]] = []  # (timestamp, index, x, y, label)
        fm_items: List[Tuple[int, int, float, float, str]] = []

        for i, label in enumerate(labels):
            name = label.split('.')[0] if '.' in label else label

            is_f0 = bool(f0_pattern.search(name))
            is_fm = bool(fm_pattern.search(name))

            # Skip anything that is not F0 and not Fm (e.g. OJIP frames)
            if not is_f0 and not is_fm:
                continue

            parts = name.split('_')
            timestamp = None
            for part in reversed(parts):
                if part.isdigit():
                    timestamp = int(part)
                    break

            if timestamp is None:
                timestamp = i

            if is_f0:
                f0_items.append((timestamp, i, x_list[i], y_list[i], label))
            elif is_fm:
                fm_items.append((timestamp, i, x_list[i], y_list[i], label))

        if not f0_items or not fm_items:
            return None

        # Sort by timestamp, then by index
        f0_items.sort(key=lambda item: (item[0], item[1]))
        fm_items.sort(key=lambda item: (item[0], item[1]))

        out_x: List[float] = []
        out_y: List[float] = []
        out_labels: List[str] = []

        used_fm_indices = set()

        for f0_ts, f0_idx, f0_x, f0_y, f0_lbl in f0_items:
            best_match = None
            best_dist = float('inf')

            for fm_ts, fm_idx, fm_x, fm_y, fm_lbl in fm_items:
                if fm_idx in used_fm_indices:
                    continue
                dist = abs(fm_ts - f0_ts)
                if dist < best_dist:
                    best_dist = dist
                    best_match = (fm_idx, fm_x, fm_y, fm_lbl)

            if best_match is not None:
                fm_idx, fm_x, fm_y, fm_lbl = best_match
                used_fm_indices.add(fm_idx)

                phi = (fm_y - f0_y) / fm_y if fm_y != 0 else 0.0
                out_x.append(fm_x)
                out_y.append(phi)
                out_labels.append(f"{f0_lbl}+{fm_lbl}")

        if not out_x:
            return None

        return out_x, out_y, out_labels

    def _process_retr(
        self,
        x_list: List[float],
        y_list: List[float],
        labels: Optional[List[str]] = None
    ) -> Tuple[List[float], List[float], Optional[List[str]]]:
        """Process TARA sequence into rETR curve."""
        blocks = self.parse_blocks(self.retr_blocks)
        if not blocks:
            blocks = self.parse_blocks(self.DEFAULT_BLOCKS)

        out_x: List[float] = []
        out_y: List[float] = []
        out_labels: List[str] = []

        total_pts = len(y_list)

        for block_idx, (start, end) in enumerate(blocks):
            if start >= total_pts:
                break

            actual_end = min(end, total_pts - 1)
            b_y = y_list[start : actual_end + 1]
            b_x = x_list[start : actual_end + 1]
            b_labels = labels[start : actual_end + 1] if labels else []

            if not b_y:
                continue

            yields: List[float] = []
            yield_labels: List[str] = []
            yield_x: List[float] = []

            if self.retr_pair_f0_fm:
                num_pairs = len(b_y) // 2
                for k in range(num_pairs):
                    f0 = b_y[2 * k]
                    fm = b_y[2 * k + 1]
                    y_val = (fm - f0) / fm if fm != 0 else 0.0
                    yields.append(y_val)
                    yield_x.append(b_x[2 * k + 1])
                    if b_labels and 2 * k + 1 < len(b_labels):
                        yield_labels.append(f"{b_labels[2 * k]}+{b_labels[2 * k + 1]}")
                    elif b_labels and 2 * k < len(b_labels):
                        yield_labels.append(b_labels[2 * k])
            else:
                yields = list(b_y)
                yield_x = list(b_x)
                yield_labels = list(b_labels)

            if not yields:
                continue

            intensity = self.get_actinic_intensity(b_labels, block_idx, b_x[0])

            if self.retr_mode == "Average":
                selected_yield = sum(yields) / len(yields)
                retr_val = selected_yield * intensity
                intensity_str = f"{int(intensity)}" if intensity.is_integer() else f"{intensity}"
                selected_label = f"Palier_{block_idx + 1}_A{intensity_str}_avg"
            else:
                idx = self.retr_index
                if 0 <= idx < len(yields):
                    chosen_idx = idx
                elif -len(yields) <= idx < 0:
                    chosen_idx = idx
                elif idx >= len(yields):
                    chosen_idx = len(yields) - 1
                else:
                    chosen_idx = 0

                selected_yield = yields[chosen_idx]
                retr_val = selected_yield * intensity
                intensity_str = f"{int(intensity)}" if intensity.is_integer() else f"{intensity}"
                if yield_labels and chosen_idx < len(yield_labels):
                    selected_label = yield_labels[chosen_idx]
                else:
                    selected_label = f"Palier_{block_idx + 1}_A{intensity_str}_idx{idx}"

            if self.retr_x_axis == "Actinic Intensity":
                out_x.append(intensity)
            else:
                out_x.append(float(block_idx + 1))

            out_y.append(float(retr_val))
            out_labels.append(selected_label)

        res_labels = out_labels if labels else (out_labels if out_labels else None)
        return out_x, out_y, res_labels

    def _process_npq(
        self,
        x_list: List[float],
        y_list: List[float],
        labels: Optional[List[str]] = None
    ) -> Tuple[List[float], List[float], Optional[List[str]]]:
        """
        Compute NPQ or qN.
        - 'All Data': Identical to existing NPQProcessor, filters all Fm points in the sequence.
        - 'Last Lightstep': Takes initial dark Fm, the end of each lightstep (137, 209, ...),
          and Fm points from the dark relaxation phase (559-581).
        """
        total_pts = len(y_list)
        fm_pattern = re.compile(r'Fm', re.IGNORECASE)
        f0_pattern = re.compile(r'F0', re.IGNORECASE)

        def is_fm_point(idx: int) -> bool:
            if labels and idx < len(labels):
                lbl = labels[idx]
                # Label has Fm and is not an F0+... pair where F0 is first
                if fm_pattern.search(lbl):
                    # If label has both F0 and Fm (like F0+Fm), it represents a pair with Fm
                    return True
                if f0_pattern.search(lbl):
                    return False
            # Fallback by parity: odd indices are Fm (1, 3, 127... 137, 201... 209)
            return (idx % 2 != 0)

        if self.npq_mode == "All Data":
            filtered_indices: List[int] = []
            for i in range(total_pts):
                if is_fm_point(i):
                    filtered_indices.append(i)

            if not filtered_indices:
                return [], [], []

            if self.npq_use_max_fm:
                fm_ref = max(y_list[i] for i in filtered_indices)
            else:
                ref_i = max(0, min(self.npq_initial_fm_index, len(filtered_indices) - 1))
                fm_ref = y_list[filtered_indices[ref_i]]

            out_x: List[float] = []
            out_y: List[float] = []
            out_labels: List[str] = []

            for i in filtered_indices:
                y = y_list[i]
                if self.npq_calc_type == "qN":
                    val = (fm_ref - y) / fm_ref if fm_ref != 0 else 0.0
                else:
                    val = (fm_ref - y) / y if y != 0 else 0.0
                out_x.append(x_list[i])
                out_y.append(val)
                lbl = labels[i] if (labels and i < len(labels)) else f"Point_{i}"
                out_labels.append(lbl)

            return out_x, out_y, out_labels if labels else None

        else:
            # 'Last Lightstep' mode
            selected_indices: List[int] = []

            # 1. Initial dark Fm
            init_idx = self.npq_initial_fm_index
            if init_idx >= total_pts:
                init_idx = 1 if total_pts > 1 else 0
            selected_indices.append(init_idx)

            # 2. End of each lightstep block (index 137, 209, 281, 344-353 -> 353, 416-425 -> 425, 488-497 -> 497)
            blocks = self.parse_blocks(self.retr_blocks)
            if not blocks:
                blocks = self.parse_blocks(self.DEFAULT_BLOCKS)

            for s, e in blocks:
                if e < total_pts:
                    # e is typically the last point of the block (an Fm)
                    target_fm = e if is_fm_point(e) else (e - 1 if e - 1 >= s else e)
                    selected_indices.append(target_fm)
                elif s < total_pts:
                    last_pt = total_pts - 1
                    target_fm = last_pt if is_fm_point(last_pt) else (last_pt - 1 if last_pt - 1 >= s else last_pt)
                    selected_indices.append(target_fm)

            # 3. Dark relaxation points from range 559-581 (filtered for Fm)
            dark_matches = re.findall(r'(\d+)\s*[-:]\s*(\d+)', self.npq_dark_relax)
            for d_start_str, d_end_str in dark_matches:
                d_s, d_e = int(d_start_str), int(d_end_str)
                if d_s > d_e:
                    d_s, d_e = d_e, d_s
                d_e = min(d_e, total_pts - 1)
                for idx in range(d_s, d_e + 1):
                    if is_fm_point(idx):
                        selected_indices.append(idx)

            # Deduplicate preserving order
            seen = set()
            unique_indices: List[int] = []
            for idx in selected_indices:
                if idx not in seen and 0 <= idx < total_pts:
                    seen.add(idx)
                    unique_indices.append(idx)

            if not unique_indices:
                return [], [], []

            # Reference Fm: initial Fm (index unique_indices[0]) or max Fm
            if self.npq_use_max_fm:
                fm_ref = max(y_list[i] for i in unique_indices)
            else:
                fm_ref = y_list[unique_indices[0]]

            out_x = []
            out_y = []
            out_labels = []

            for i in unique_indices:
                y = y_list[i]
                if self.npq_calc_type == "qN":
                    val = (fm_ref - y) / fm_ref if fm_ref != 0 else 0.0
                else:
                    val = (fm_ref - y) / y if y != 0 else 0.0
                out_x.append(x_list[i])
                out_y.append(val)
                lbl = labels[i] if (labels and i < len(labels)) else f"Point_{i}"
                out_labels.append(lbl)

            return out_x, out_y, out_labels if labels else None

    def process(
        self,
        x_list: List[float],
        y_list: List[float],
        labels: Optional[List[str]] = None
    ) -> Union[Tuple[List[float], List[float]], Tuple[List[float], List[float], Optional[List[str]]]]:
        """Process the input data based on export_type."""
        if not self.validate_data(x_list, y_list):
            return x_list.copy(), y_list.copy(), (labels.copy() if labels else None)
        
        out_x: List[float] = []
        out_y: List[float] = []
        out_labels: List[str] = []
        
        if self.export_type == "Raw":
            return x_list.copy(), y_list.copy(), (labels.copy() if labels else None)

        elif self.export_type == "FvFm":
            # We need at least 4 points to compute 2 Fv/Fm
            if len(y_list) >= 4:
                # Fv/Fm 1: (y[1] - y[0]) / y[1]
                fm1 = y_list[1]
                f01 = y_list[0]
                fvfm1 = (fm1 - f01) / fm1 if fm1 != 0 else 0.0
                out_x.append(x_list[1])
                out_y.append(fvfm1)
                if labels and len(labels) >= 2:
                    out_labels.append(f"{labels[0]}+{labels[1]}")
                
                # Fv/Fm 2: (y[3] - y[2]) / y[3]
                fm2 = y_list[3]
                f02 = y_list[2]
                fvfm2 = (fm2 - f02) / fm2 if fm2 != 0 else 0.0
                out_x.append(x_list[3])
                out_y.append(fvfm2)
                if labels and len(labels) >= 4:
                    out_labels.append(f"{labels[2]}+{labels[3]}")

        elif self.export_type == "Phi PSII":
            # Compute Phi PSII = (Fm - F0) / Fm for ALL F0/Fm pairs,
            # using smart label filtering to skip OJIP and non-F0/Fm frames
            return self._process_phi_psii(x_list, y_list, labels)

        elif self.export_type == "OJIP":
            if labels:
                for i, label in enumerate(labels):
                    if "OJIP" in label.upper():
                        parts = label.split("_")
                        if len(parts) >= 2:
                            try:
                                t_val = float(parts[-2])
                                out_x.append(t_val)
                                out_y.append(y_list[i])
                                out_labels.append(label)
                            except ValueError:
                                pass

        elif self.export_type == "rETR":
            return self._process_retr(x_list, y_list, labels)

        elif self.export_type == "NPQ":
            return self._process_npq(x_list, y_list, labels)

        else:
            out_x = x_list.copy()
            out_y = y_list.copy()
            out_labels = labels.copy() if labels else []

        result_labels: Optional[List[str]] = out_labels if labels else None
        return out_x, out_y, result_labels

    def create_ui(self, parent: Union[int, str]) -> None:
        """Create UI controls."""
        def update_tooltip(app_data: str) -> None:
            if not hasattr(self, 'tooltip_text') or not dpg.does_item_exist(self.tooltip_text):
                return
            if app_data == "Raw":
                dpg.set_value(
                    self.tooltip_text,
                    "Raw Mode:\nPass-through mode. Outputs the input sequence\ndirectly without modification."
                )
            elif app_data == "FvFm":
                dpg.set_value(
                    self.tooltip_text, 
                    "FvFm Mode:\nCalculates two Fv/Fm ratios from the\nfirst 4 points (F0/Fm pairs)."
                )
            elif app_data == "Phi PSII":
                dpg.set_value(
                    self.tooltip_text,
                    "Phi PSII Mode:\nCalculates quantum yield (Fm - F0) / Fm\nfor ALL (F0, Fm) pairs across the entire sequence."
                )
            elif app_data == "OJIP":
                dpg.set_value(
                    self.tooltip_text, 
                    "OJIP Mode:\nExtracts fast kinetics points (tagged 'OJIP')\nand plots them against their acquisition time."
                )
            elif app_data == "rETR":
                dpg.set_value(
                    self.tooltip_text,
                    "rETR Mode:\nCalculates relative Electron Transport Rate\nacross actinic light blocks (paliers)."
                )
            elif app_data == "NPQ":
                dpg.set_value(
                    self.tooltip_text,
                    "NPQ Mode:\nCalculates Non-Photochemical Quenching (NPQ or qN)\nusing All Data or Last Lightstep (+ dark relaxation)."
                )

        def update_groups_visibility(export_type: str) -> None:
            is_retr = (export_type == "rETR")
            is_npq = (export_type == "NPQ")
            if hasattr(self, 'retr_ui_group') and dpg.does_item_exist(self.retr_ui_group):
                dpg.configure_item(self.retr_ui_group, show=is_retr)
            if hasattr(self, 'npq_ui_group') and dpg.does_item_exist(self.npq_ui_group):
                dpg.configure_item(self.npq_ui_group, show=is_npq)

        def combo_callback(sender: Union[int, str], app_data: str, user_data: Any, *args: Any, **kwargs: Any) -> None:
            self.update_setting("export_type", app_data)
            update_tooltip(app_data)
            update_groups_visibility(app_data)

        def retr_mode_callback(sender: Union[int, str], app_data: str, user_data: Any, *args: Any, **kwargs: Any) -> None:
            self.update_setting("retr_mode", app_data)
            if hasattr(self, 'retr_index_widget') and dpg.does_item_exist(self.retr_index_widget):
                dpg.configure_item(self.retr_index_widget, show=(app_data == "Index"))

        def npq_mode_callback(sender: Union[int, str], app_data: str, user_data: Any, *args: Any, **kwargs: Any) -> None:
            self.update_setting("npq_mode", app_data)
            if hasattr(self, 'npq_dark_relax_widget') and dpg.does_item_exist(self.npq_dark_relax_widget):
                dpg.configure_item(self.npq_dark_relax_widget, show=(app_data == "Last Lightstep"))

        with dpg.group(parent=parent):
            dpg.add_text("Export Data:")
            widget = dpg.add_combo(
                items=["Raw", "FvFm", "Phi PSII", "OJIP", "rETR", "NPQ"],
                default_value=self.export_type,
                callback=combo_callback,
                width=150
            )
            self.register_param("export_type", widget)
            
            with dpg.tooltip(widget):
                self.tooltip_text = dpg.add_text("", wrap=250)
                
            update_tooltip(self.export_type)

            # rETR configuration group
            self.retr_ui_group = dpg.add_group(show=(self.export_type == "rETR"))
            with dpg.group(parent=self.retr_ui_group):
                dpg.add_text("rETR Method:")
                rm_combo = dpg.add_combo(
                    items=["Average", "Index"],
                    default_value=self.retr_mode,
                    callback=retr_mode_callback,
                    width=150
                )
                self.register_param("retr_mode", rm_combo)
                with dpg.tooltip(rm_combo):
                    dpg.add_text(
                        "Average: computes mean yield of all flashes in each palier.\n"
                        "Index: selects one specific flash per palier (e.g. -1 for last, 2 for 3rd).",
                        wrap=250
                    )

                self.retr_index_widget = dpg.add_input_int(
                    label="Point Index",
                    default_value=self.retr_index,
                    callback=lambda s, a, u: self.update_setting("retr_index", int(a)),
                    width=100,
                    show=(self.retr_mode == "Index")
                )
                self.register_param("retr_index", self.retr_index_widget)
                with dpg.tooltip(self.retr_index_widget):
                    dpg.add_text(
                        "Index of flash within palier:\n"
                        "-1 = last measurement (steady state)\n"
                        "0 = 1st measurement\n"
                        "2 = 3rd measurement",
                        wrap=250
                    )

                pair_cb = dpg.add_checkbox(
                    label="Pair (F0, Fm)",
                    default_value=self.retr_pair_f0_fm,
                    callback=lambda s, a, u: self.update_setting("retr_pair_f0_fm", bool(a))
                )
                self.register_param("retr_pair_f0_fm", pair_cb)
                with dpg.tooltip(pair_cb):
                    dpg.add_text(
                        "When enabled, pairs consecutive points (F0, Fm) in each block\n"
                        "to compute yield = (Fm - F0) / Fm before multiplying by intensity.",
                        wrap=250
                    )

                dpg.add_text("X Axis:")
                xaxis_combo = dpg.add_combo(
                    items=["Actinic Intensity", "Palier Index"],
                    default_value=self.retr_x_axis,
                    callback=lambda s, a, u: self.update_setting("retr_x_axis", str(a)),
                    width=150
                )
                self.register_param("retr_x_axis", xaxis_combo)

                dpg.add_text("Blocks (F0/Fm ranges):")
                blocks_inp = dpg.add_input_text(
                    default_value=self.retr_blocks,
                    callback=lambda s, a, u: self.update_setting("retr_blocks", str(a)),
                    width=200
                )
                self.register_param("retr_blocks", blocks_inp)
                with dpg.tooltip(blocks_inp):
                    dpg.add_text(
                        "Comma-separated start-end index ranges for each actinic level.\n"
                        "Default: 126-137, 200-209, 272-281, 344-353, 416-425, 488-497",
                        wrap=250
                    )

            # NPQ configuration group
            self.npq_ui_group = dpg.add_group(show=(self.export_type == "NPQ"))
            with dpg.group(parent=self.npq_ui_group):
                dpg.add_text("NPQ Selection:")
                npq_mode_combo = dpg.add_combo(
                    items=["Last Lightstep", "All Data"],
                    default_value=self.npq_mode,
                    callback=npq_mode_callback,
                    width=150
                )
                self.register_param("npq_mode", npq_mode_combo)
                with dpg.tooltip(npq_mode_combo):
                    dpg.add_text(
                        "Last Lightstep: Initial dark Fm + end of each lightstep (index 137, 209, ...) + dark relaxation (559-581).\n"
                        "All Data: Filters all Fm points in the sequence (identical to standard NPQ processor).",
                        wrap=250
                    )

                dpg.add_text("Calculation:")
                calc_combo = dpg.add_combo(
                    items=["NPQ", "qN"],
                    default_value=self.npq_calc_type,
                    callback=lambda s, a, u: self.update_setting("npq_calc_type", str(a)),
                    width=150
                )
                self.register_param("npq_calc_type", calc_combo)
                with dpg.tooltip(calc_combo):
                    dpg.add_text("Formula:\nNPQ = (Fm_ref - Fm') / Fm'\nqN = (Fm_ref - Fm') / Fm_ref", wrap=250)

                max_fm_cb = dpg.add_checkbox(
                    label="Use Max Fm as Ref",
                    default_value=self.npq_use_max_fm,
                    callback=lambda s, a, u: self.update_setting("npq_use_max_fm", bool(a))
                )
                self.register_param("npq_use_max_fm", max_fm_cb)
                with dpg.tooltip(max_fm_cb):
                    dpg.add_text("If checked, uses the maximum Fm value detected as Fm_ref instead of initial dark Fm.", wrap=250)

                init_fm_inp = dpg.add_input_int(
                    label="Initial Fm Index",
                    default_value=self.npq_initial_fm_index,
                    callback=lambda s, a, u: self.update_setting("npq_initial_fm_index", int(a)),
                    width=100
                )
                self.register_param("npq_initial_fm_index", init_fm_inp)
                with dpg.tooltip(init_fm_inp):
                    dpg.add_text("Index of the dark initial Fm reference (usually 1 or 3).", wrap=250)

                self.npq_dark_relax_widget = dpg.add_input_text(
                    label="Dark Relax Range",
                    default_value=self.npq_dark_relax,
                    callback=lambda s, a, u: self.update_setting("npq_dark_relax", str(a)),
                    width=150,
                    show=(self.npq_mode == "Last Lightstep")
                )
                self.register_param("npq_dark_relax", self.npq_dark_relax_widget)
                with dpg.tooltip(self.npq_dark_relax_widget):
                    dpg.add_text("Index range for dark relaxation recovery (e.g. 559-581). Fm points are automatically filtered.", wrap=250)

    def set_parameters(self, params: Dict[str, Any]) -> None:
        super().set_parameters(params)
        if hasattr(self, 'tooltip_text') and dpg.does_item_exist(self.tooltip_text):
            if self.export_type == "Raw":
                dpg.set_value(
                    self.tooltip_text,
                    "Raw Mode:\nPass-through mode. Outputs the input sequence\ndirectly without modification."
                )
            elif self.export_type == "FvFm":
                dpg.set_value(
                    self.tooltip_text, 
                    "FvFm Mode:\nCalculates two Fv/Fm ratios from the\nfirst 4 points (F0/Fm pairs)."
                )
            elif self.export_type == "Phi PSII":
                dpg.set_value(
                    self.tooltip_text,
                    "Phi PSII Mode:\nCalculates quantum yield (Fm - F0) / Fm\nfor ALL (F0, Fm) pairs across the entire sequence."
                )
            elif self.export_type == "OJIP":
                dpg.set_value(
                    self.tooltip_text, 
                    "OJIP Mode:\nExtracts fast kinetics points (tagged 'OJIP')\nand plots them against their acquisition time."
                )
            elif self.export_type == "rETR":
                dpg.set_value(
                    self.tooltip_text,
                    "rETR Mode:\nCalculates relative Electron Transport Rate\nacross actinic light blocks (paliers)."
                )
            elif self.export_type == "NPQ":
                dpg.set_value(
                    self.tooltip_text,
                    "NPQ Mode:\nCalculates Non-Photochemical Quenching (NPQ or qN)\nusing All Data or Last Lightstep (+ dark relaxation)."
                )
        if hasattr(self, 'retr_ui_group') and dpg.does_item_exist(self.retr_ui_group):
            dpg.configure_item(self.retr_ui_group, show=(self.export_type == "rETR"))
        if hasattr(self, 'retr_index_widget') and dpg.does_item_exist(self.retr_index_widget):
            dpg.configure_item(self.retr_index_widget, show=(self.retr_mode == "Index"))
        if hasattr(self, 'npq_ui_group') and dpg.does_item_exist(self.npq_ui_group):
            dpg.configure_item(self.npq_ui_group, show=(self.export_type == "NPQ"))
        if hasattr(self, 'npq_dark_relax_widget') and dpg.does_item_exist(self.npq_dark_relax_widget):
            dpg.configure_item(self.npq_dark_relax_widget, show=(self.npq_mode == "Last Lightstep"))
