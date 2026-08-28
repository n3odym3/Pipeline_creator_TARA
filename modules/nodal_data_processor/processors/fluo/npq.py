import re
from typing import List, Tuple, Optional, Union
import dearpygui.dearpygui as dpg
from modules.nodal_data_processor.data_processor_base import DataProcessorBase


class NPQProcessor(DataProcessorBase):
    """Compute NPQ or qN. Formulas: qN = (Fm - Fm') / Fm, NPQ = (Fm - Fm') / Fm'. Filters for labels containing 'Fm'."""

    def __init__(self, node_id: Union[int, str]):
        super().__init__(node_id)
        self.use_max_fm: bool = False
        self.ref_index: int = 0
        self.calc_type: str = "qN"

    def process(self, x_list: List[float], y_list: List[float], labels: Optional[List[str]] = None) -> Tuple[List[float], List[float], Optional[List[str]]]:
        """
        Compute calculation based on a reference Fm.
        qN Formula: (Fm_ref - Fm_current) / Fm_ref
        NPQ Formula: (Fm_ref - Fm_current) / Fm_current
        """
        if not self.validate_data(x_list, y_list) or not labels:
            return x_list, y_list, labels

        # 1. Filter data to keep only those with "Fm" in labels
        fm_pattern = re.compile(r'Fm', re.IGNORECASE)
        filtered_data: List[Tuple[float, float, str]] = [] # List of (x, y, label)
        
        for i, label in enumerate(labels):
            if fm_pattern.search(label):
                filtered_data.append((x_list[i], y_list[i], label))
        
        if not filtered_data:
            return [], [], []

        # 2. Determine Fm_ref
        if self.use_max_fm:
            # Find max Y in filtered data
            fm_ref = max(d[1] for d in filtered_data)
        else:
            # Use specific index in filtered data
            idx = max(0, min(self.ref_index, len(filtered_data) - 1))
            fm_ref = filtered_data[idx][1]

        # 3. Calculate selected parameter for each filtered point
        out_x = []
        out_y = []
        out_labels = []
        
        for x, y, label in filtered_data:
            if self.calc_type == "qN":
                npq = (fm_ref - y) / fm_ref if fm_ref != 0 else 0.0
            else: # "NPQ"
                npq = (fm_ref - y) / y if y != 0 else 0.0
            out_x.append(x)
            out_y.append(npq)
            out_labels.append(label)

        return out_x, out_y, out_labels

    def create_ui(self, parent: Union[int, str]) -> None:
        """Create UI controls."""
        max_check = dpg.add_checkbox(
            label="Use Max Fm as Ref",
            default_value=self.use_max_fm,
            callback=lambda s, v: setattr(self, "use_max_fm", v),
            parent=parent
        )
        self.register_param("use_max_fm", max_check)
        with dpg.tooltip(max_check):
            dpg.add_text("If checked, NPQ ref Fm is the maximum Fm value detected.", wrap=250)

        idx_input = dpg.add_input_int(
            label="Ref Index",
            default_value=self.ref_index,
            callback=lambda s, v: setattr(self, "ref_index", v),
            parent=parent,
            width=100
        )
        self.register_param("ref_index", idx_input)
        with dpg.tooltip(idx_input):
            dpg.add_text("Index of the point to use as Fm reference when Max Fm is unchecked.", wrap=250)

        calc_type_combo = dpg.add_combo(
            items=["qN", "NPQ"],
            default_value=self.calc_type,
            label="Calc Type",
            callback=lambda s, v: setattr(self, "calc_type", v),
            parent=parent,
            width=100
        )
        self.register_param("calc_type", calc_type_combo)
        with dpg.tooltip(calc_type_combo):
            dpg.add_text("Choose calculation formula:\nqN: (Fm - Fm') / Fm\nNPQ: (Fm - Fm') / Fm'", wrap=250)



