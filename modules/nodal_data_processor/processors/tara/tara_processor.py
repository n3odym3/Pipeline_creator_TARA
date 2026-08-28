from typing import List, Tuple, Optional, Union, Dict, Any
import dearpygui.dearpygui as dpg

from modules.nodal_data_processor.data_processor_base import DataProcessorBase

class TaraProcessor(DataProcessorBase):
    """Automatically parses and splits sequences from TARA experiments."""

    def __init__(self, node_id: Union[int, str]):
        super().__init__(node_id)
        
        # Define exporter parameter
        self.export_type: str = "FvFm"
        self.tooltip_text: Union[int, str] = -1
        
        # Override name for display
        self.name: str = "TARA Exporter"

    def process(self, x_list: List[float], y_list: List[float], labels: Optional[List[str]] = None) -> Union[Tuple[List[float], List[float]], Tuple[List[float], List[float], Optional[List[str]]]]:
        """Process the input data based on export_type."""
        if not self.validate_data(x_list, y_list):
            return x_list.copy(), y_list.copy(), (labels.copy() if labels else None)
        
        out_x = []
        out_y = []
        out_labels = []
        
        if self.export_type == "FvFm":
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
                
                # Keep other labels empty if necessary, but 2 points means 2 labels.
            else:
                # Not enough points, return empty
                pass

        elif self.export_type == "OJIP":
            if labels:
                for i, label in enumerate(labels):
                    if "OJIP" in label.upper() or "OJIP" in label:
                        parts = label.split("_")
                        if len(parts) >= 2:
                            try:
                                t_val = float(parts[-2])
                                out_x.append(t_val)
                                out_y.append(y_list[i])
                                out_labels.append(label)
                            except ValueError:
                                pass

        else:
            # Fallback
            out_x = x_list.copy()
            out_y = y_list.copy()
            out_labels = labels.copy() if labels else None

        result_labels = out_labels if labels else None
        return out_x, out_y, result_labels

    def create_ui(self, parent: Union[int, str]) -> None:
        """Create UI controls."""
        def update_tooltip(app_data: str) -> None:
            if not hasattr(self, 'tooltip_text') or not dpg.does_item_exist(self.tooltip_text):
                return
            if app_data == "FvFm":
                dpg.set_value(
                    self.tooltip_text, 
                    "FvFm Mode:\nCalculates two Fv/Fm ratios from the\nfirst 4 points (F0/Fm pairs)."
                )
            elif app_data == "OJIP":
                dpg.set_value(
                    self.tooltip_text, 
                    "OJIP Mode:\nExtracts fast kinetics points (tagged 'OJIP')\nand plots them against their acquisition time."
                )

        def combo_callback(sender: Union[int, str], app_data: str, user_data: Any, *args: Any, **kwargs: Any) -> None:
            setattr(self, "export_type", app_data)
            update_tooltip(app_data)

        with dpg.group(parent=parent):
            dpg.add_text("Export Data:")
            widget = dpg.add_combo(
                items=["FvFm", "OJIP"],
                default_value=self.export_type,
                callback=combo_callback,
                width=150
            )
            self.register_param("export_type", widget)
            
            with dpg.tooltip(widget):
                self.tooltip_text = dpg.add_text("", wrap=250)
                
            update_tooltip(self.export_type)

    def set_parameters(self, params: Dict[str, Any]) -> None:
        super().set_parameters(params)
        if hasattr(self, 'tooltip_text') and dpg.does_item_exist(self.tooltip_text):
            if self.export_type == "FvFm":
                dpg.set_value(
                    self.tooltip_text, 
                    "FvFm Mode:\nCalculates two Fv/Fm ratios from the\nfirst 4 points (F0/Fm pairs)."
                )
            elif self.export_type == "OJIP":
                dpg.set_value(
                    self.tooltip_text, 
                    "OJIP Mode:\nExtracts fast kinetics points (tagged 'OJIP')\nand plots them against their acquisition time."
                )


