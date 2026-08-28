import re
from typing import List, Tuple, Optional, Union, Dict, Any
import dearpygui.dearpygui as dpg

from modules.nodal_data_processor.data_processor_base import DataProcessorBase

class PhiProcessor(DataProcessorBase):
    """Compute Phi PSII, Phi NPQ, and Phi NO from Fs and Fm' points. 
    Requires a valid Fm reference."""

    def __init__(self, node_id: Union[int, str]):
        super().__init__(node_id)
        
        self.export_mode: str = "Phi PSII"
        self.use_auto_ref: bool = True
        self.f0_ref_index: int = 0
        self.fm_ref_index: int = 1
        self.tooltip_text: Union[int, str] = -1
        
        self.name: str = "Phi Calculator"
        self.auxiliary_outputs: List[str] = ["PSII", "NPQ", "NO"]

    def process(self, x_list: List[float], y_list: List[float], labels: Optional[List[str]] = None) -> Union[Tuple[List[float], List[float], Optional[List[str]]], Dict[str, Any]]:
        """
        Compute chosen parameter:
        Phi_PSII = (Fm' - Fs) / Fm'
        Phi_NPQ = (Fs / Fm') - (Fs / Fm)
        Phi_NO = Fs / Fm
        ALL 3 = Computes and outputs all three.
        """
        if self.export_mode == "ALL 3":
            # Compute PSII
            self.export_mode = "Phi PSII"
            psii_x, psii_y, psii_labels = self.process(x_list, y_list, labels)
            
            # Compute NPQ
            self.export_mode = "Phi NPQ"
            npq_x, npq_y, npq_labels = self.process(x_list, y_list, labels)
            
            # Compute NO
            self.export_mode = "Phi NO"
            no_x, no_y, no_labels = self.process(x_list, y_list, labels)
            
            # Restore export_mode
            self.export_mode = "ALL 3"
            
            return {
                "Out": (psii_x, psii_y, psii_labels),
                "__skip_main_output__": True,
                "PSII": {
                    "x": psii_x,
                    "y": psii_y,
                    "labels": psii_labels,
                    "name": "_PSII"
                },
                "NPQ": {
                    "x": npq_x,
                    "y": npq_y,
                    "labels": npq_labels,
                    "name": "_NPQ"
                },
                "NO": {
                    "x": no_x,
                    "y": no_y,
                    "labels": no_labels,
                    "name": "_NO"
                }
            }

        if not self.validate_data(x_list, y_list):
            return x_list.copy(), y_list.copy(), labels.copy() if labels else None

        # Determine reference F0 and Fm indices
        if self.use_auto_ref:
            f0 = y_list[0] if y_list else 1.0
            fm = max(y_list) if y_list else 1.0
        else:
            f0 = y_list[self.f0_ref_index] if 0 <= self.f0_ref_index < len(y_list) else (y_list[0] if y_list else 1.0)
            fm = y_list[self.fm_ref_index] if 0 <= self.fm_ref_index < len(y_list) else (max(y_list) if y_list else 1.0)

        if fm <= 0:
            fm = 1e-9 # Prevent div by zero if strange data

        out_x = []
        out_y = []
        out_labels = []

        if labels and len(labels) == len(y_list):
            # Smart pairing based on labels
            fs_data: Dict[int, Tuple[float, float, str]] = {} # {timestamp: (x, y, label)}
            fm_prime_data: Dict[int, Tuple[float, float, str]] = {}
            
            # The user explicitly wants labeled 'F0' to act as Fs.
            fs_pattern = re.compile(r'(?:^|[_\-])(?:Fs|F|F0)(?:[_\-]|$)', re.IGNORECASE)
            fm_prime_pattern = re.compile(r'(?:^|[_\-])Fm(?:p|\')?(?:[_\-]|$)', re.IGNORECASE)
            
            for i, label in enumerate(labels):
                name = label.split('.')[0] if '.' in label else label
                
                is_fs = bool(fs_pattern.search(name))
                is_fm_prime = bool(fm_prime_pattern.search(name))
                
                if not is_fs and not is_fm_prime:
                    continue
                
                parts = name.split('_')
                timestamp = i
                for part in reversed(parts):
                    if part.isdigit():
                        timestamp = int(part)
                        break
                        
                if is_fs:
                    fs_data[timestamp] = (x_list[i], y_list[i], label)
                elif is_fm_prime:
                    fm_prime_data[timestamp] = (x_list[i], y_list[i], label)

            fs_ts = sorted(fs_data.keys())
            fm_ts = sorted(fm_prime_data.keys())
            
            used_fm = set()
            paired = False
            for ts in fs_ts:
                best_fm_ts = None
                best_dist = float('inf')
                
                for fmp_ts in fm_ts:
                    if fmp_ts in used_fm:
                        continue
                    if fmp_ts >= ts:
                        dist = fmp_ts - ts
                        if dist < best_dist:
                            best_dist = dist
                            best_fm_ts = fmp_ts

                if best_fm_ts is not None:
                    used_fm.add(best_fm_ts)
                    fs_x, fs_y, fs_label = fs_data[ts]
                    fmp_x, fmp_y, fmp_label = fm_prime_data[best_fm_ts]

                    # Standard formulations don't directly use F0, but if you wanted to implement Kramer using F0, you could use `f0`. Here we just need `fm`.
                    val = self._compute_phi(fs_y, fmp_y, fm)
                    out_x.append(fmp_x)
                    out_y.append(val)
                    out_labels.append(f"{fs_label}+{fmp_label}")
                    paired = True
                    
            if paired:
                return out_x, out_y, out_labels

        # Fallback sequential pairing
        return self._process_sequential(x_list, y_list, labels, fm)

    def _process_sequential(self, x_list: List[float], y_list: List[float], labels: Optional[List[str]], fm: float) -> Tuple[List[float], List[float], Optional[List[str]]]:
        out_x = []
        out_y = []
        out_labels = []
        # sequential pairing: Fs, Fm', Fs, Fm'
        # we assume pairs are (y[i], y[i+1])
        for i in range(0, len(y_list) - 1, 2):
            fs = y_list[i]
            fmp = y_list[i+1]
            val = self._compute_phi(fs, fmp, fm)
            out_x.append(x_list[i+1])
            out_y.append(val)
            if labels and i+1 < len(labels):
                out_labels.append(f"{labels[i]}+{labels[i+1]}")
                
        result_labels = out_labels if out_labels else labels
        return out_x, out_y, result_labels
        
    def _compute_phi(self, fs: float, fmp: float, fm: float) -> float:
        if fmp <= 0: return 0.0
        
        if self.export_mode == "Phi PSII":
            return (fmp - fs) / fmp
        elif self.export_mode == "Phi NPQ":
            return (fs / fmp) - (fs / fm)
        elif self.export_mode == "Phi NO":
            return fs / fm
        return 0.0

    def create_ui(self, parent: Union[int, str]) -> None:
        def update_tooltip(app_data: str) -> None:
            if not hasattr(self, 'tooltip_text') or not dpg.does_item_exist(self.tooltip_text):
                return
            if app_data == "Phi PSII":
                dpg.set_value(
                    self.tooltip_text, 
                    "Phi PSII Mode:\n(Fm' - Fs) / Fm'\nMeasures the quantum yield of PSII photochemistry."
                )
            elif app_data == "Phi NPQ":
                dpg.set_value(
                    self.tooltip_text, 
                    "Phi NPQ Mode:\n(Fs / Fm') - (Fs / Fm)\nMeasures the quantum yield of non-photochemical quenching."
                )
            elif app_data == "Phi NO":
                dpg.set_value(
                    self.tooltip_text, 
                    "Phi NO Mode:\nFs / Fm\nMeasures the quantum yield of non-regulated energy dissipation."
                )
            elif app_data == "ALL 3":
                dpg.set_value(
                    self.tooltip_text, 
                    "ALL 3 Mode:\nCalculates and outputs all 3 parameters (Phi PSII, Phi NPQ, and Phi NO) as separate samples with suffixes."
                )

        def combo_callback(sender: Union[int, str], app_data: str, user_data: Any, *args: Any, **kwargs: Any) -> None:
            setattr(self, "export_mode", app_data)
            update_tooltip(app_data)

        with dpg.group(parent=parent):
            combo = dpg.add_combo(
                items=["Phi PSII", "Phi NPQ", "Phi NO", "ALL 3"],
                default_value=self.export_mode,
                callback=combo_callback,
                width=150
            )
            self.register_param("export_mode", combo)
            
            with dpg.tooltip(combo):
                self.tooltip_text = dpg.add_text("", wrap=250)
                
            update_tooltip(self.export_mode)
                
            auto_ref = dpg.add_checkbox(
                label="Auto F0/Fm",
                default_value=self.use_auto_ref,
                callback=lambda s, a, u: setattr(self, "use_auto_ref", a)
            )
            self.register_param("use_auto_ref", auto_ref)
            with dpg.tooltip(auto_ref):
                dpg.add_text("Auto: F0 is 1st point, Fm is global max.", wrap=250)
            
            f0_input = dpg.add_input_int(
                label="F0 Index",
                default_value=self.f0_ref_index,
                min_value=0,
                min_clamped=True,
                callback=lambda s, a, u: setattr(self, "f0_ref_index", a),
                width=100
            )
            self.register_param("f0_ref_index", f0_input)
            with dpg.tooltip(f0_input):
                dpg.add_text("Absolute index in raw data array for F0.", wrap=250)
                
            fm_input = dpg.add_input_int(
                label="Fm Index",
                default_value=self.fm_ref_index,
                min_value=0,
                min_clamped=True,
                callback=lambda s, a, u: setattr(self, "fm_ref_index", a),
                width=100
            )
            self.register_param("fm_ref_index", fm_input)
            with dpg.tooltip(fm_input):
                dpg.add_text("Absolute index in raw data array for Fm.", wrap=250)

    def set_parameters(self, params: Dict[str, Any]) -> None:
        super().set_parameters(params)
        if hasattr(self, 'tooltip_text') and dpg.does_item_exist(self.tooltip_text):
            if self.export_mode == "Phi PSII":
                dpg.set_value(
                    self.tooltip_text, 
                    "Phi PSII Mode:\n(Fm' - Fs) / Fm'\nMeasures the quantum yield of PSII photochemistry."
                )
            elif self.export_mode == "Phi NPQ":
                dpg.set_value(
                    self.tooltip_text, 
                    "Phi NPQ Mode:\n(Fs / Fm') - (Fs / Fm)\nMeasures the quantum yield of non-photochemical quenching."
                )
            elif self.export_mode == "Phi NO":
                dpg.set_value(
                    self.tooltip_text, 
                    "Phi NO Mode:\nFs / Fm\nMeasures the quantum yield of non-regulated energy dissipation."
                )
            elif self.export_mode == "ALL 3":
                dpg.set_value(
                    self.tooltip_text, 
                    "ALL 3 Mode:\nCalculates and outputs all 3 parameters (Phi PSII, Phi NPQ, and Phi NO) as separate samples with suffixes."
                )

