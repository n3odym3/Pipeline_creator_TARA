from __future__ import annotations

import csv
import os
import threading
from typing import Any

import dearpygui.dearpygui as dpg
from loguru import logger

from core.app_state import app_state
from core.input_output_types import IOTypes
from core.window_base import WindowBase


class TaraAggregatorWin(WindowBase):
    """
    TARA Aggregator Module.
    Receives metadata from TARA and samples from Nodal Data Processor.
    Aggregates them into a table.
    """

    # Instance variables
    accepted_input_types: list[IOTypes]
    outputs: dict[str, IOTypes]
    connections: dict[str, list]
    _persistent_fields: list[str]
    current_metadata: dict[str, Any]
    metadata_keys: list[str]
    editable_fields: dict[str, bool]
    fill_with_data_key: str
    optional_fields: list[str]
    rows: list[dict[str, Any]]
    default_save_path: str
    table_container_tag: str
    table_tag: str
    replace_combo_tag: str
    clear_btn_tag: str
    _rebuild_timer: threading.Timer | None

    def __init__(
        self,
        label: str = "TARA Aggregator",
        win_width: int = 600,
        win_height: int = 400,
        pos: tuple[int, int] = (50, 50),
        outputs: dict[str, IOTypes] | None = None,
        uuid: int | None = None,
        visible: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            label=label,
            pos=pos,
            win_width=win_width,
            win_height=win_height,
            uuid=uuid,
            outputs=outputs or {},
            visible=visible,
            **kwargs,
        )

        self.accepted_input_types = [IOTypes.CMD_DICT, IOTypes.SAMPLE, IOTypes.FOLDER_PATH, IOTypes.ANY]
        self.outputs = {}  # No outputs
        self.connections = {k: [] for k in self.outputs}
        self._persistent_fields = ["label"]

        self.current_metadata = {}
        self.metadata_keys = []
        self.editable_fields = {}
        self.fill_with_data_key = "PAM_value"
        self.optional_fields = []
        self.rows = []
        self.default_save_path = ""

        self.table_container_tag = f"agreg_table_cont_{self.UUID}"
        self.table_tag = f"agreg_table_{self.UUID}"
        self.replace_combo_tag = f"agreg_replace_combo_{self.UUID}"
        self.clear_btn_tag = f"agreg_clear_{self.UUID}"
        self._rebuild_timer = None

        # Build UI
        with dpg.window(
            label=self.label,
            width=self.win_width,
            height=self.win_height,
            pos=self.pos,
            tag=self.winID,
            show=self.visible,
        ):
            with dpg.group(horizontal=True):
                dpg.add_text("Use sample name to populate:")
                dpg.add_combo(["None"], default_value="None", tag=self.replace_combo_tag, width=200)
                dpg.add_button(label="Clear Table", tag=self.clear_btn_tag, callback=self._clear_table)
                dpg.add_button(label="Export CSV", callback=self._export_csv)

            dpg.add_separator()

            with dpg.child_window(tag=self.table_container_tag, border=False):
                pass

        self._rebuild_table()
        self.update_permission()

    def update_permission(self) -> None:
        """Adjust module permissions and UI visibility based on app_state.mode."""
        is_user = app_state.mode == "user"

        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

        # Trigger table rebuild to apply cell readonly state
        self._request_rebuild()

    def input_cb(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Handle incoming data (metadata dict, sample, or folder path)."""
        data_val = kwargs.get("sample", kwargs.get("data", data))
        data_type = kwargs.get("data_type", None)

        if data_type == IOTypes.FOLDER_PATH or (isinstance(data_val, str) and os.path.isdir(data_val)):
            self.default_save_path = data_val
            logger.info(f"{self.label} set default save path: {self.default_save_path}")
            return

        if data_type == IOTypes.CMD_DICT or (isinstance(data_val, dict) and "metadata" in data_val):
            if isinstance(data_val, dict) and "metadata" in data_val:
                # This is metadata from TARA

                self.current_metadata = data_val["metadata"]

                new_keys = list(self.current_metadata.keys())
                self.metadata_keys = new_keys
                self.editable_fields = data_val.get("editable", {})
                self.fill_with_data_key = data_val.get("fill_with_data_key", "PAM_value")
                self.optional_fields = data_val.get("optional_fields", [])
                populate_default = data_val.get("populate_default", "None")
                self._update_combo_choices(default_populate=populate_default)
                self._request_rebuild()

            elif isinstance(data_val, dict) and data_val.get("clear", False):
                self._clear_table()
            elif isinstance(data_val, dict) and data_val.get("save", False):
                self._export_csv()

        elif data_type == IOTypes.SAMPLE or (isinstance(data_val, dict) and "y" in data_val):
            # This is a sample
            if not self.metadata_keys:
                msg = f"{self.label} received a sample but no metadata template has been set yet."
                logger.warning(msg)
                if dpg.does_item_exist(self.winID) and dpg.is_item_shown(self.winID):
                    from core.tutorial_manager import tutorial_manager
                    tutorial_manager.play_center_hint(msg, type="error_hint", icon_type="warning")
                return

            sample_name = data_val.get("name", "UnknownSample")
            action = data_val.get("action", "select")
            sample_uuid = data_val.get("uuid", sample_name)

            if action in ["highlight", "unhighlight", "clear"] or data_val.get("clear"):
                return

            if action == "rename":
                found = False
                for r in self.rows:
                    if r.get("_internal_uuid") == sample_uuid:
                        if dpg.does_item_exist(self.replace_combo_tag):
                            replace_field = dpg.get_value(self.replace_combo_tag)
                            if replace_field and replace_field != "None" and replace_field in r:
                                r[replace_field] = sample_name
                        found = True
                        break
                if found:
                    self._request_rebuild()
                return

            # Remove any existing row with the same UUID (useful for updates or unselects)
            self.rows = [r for r in self.rows if r.get("_internal_uuid") != sample_uuid]

            if action in ["unselect", "delete"]:
                logger.info(f"{self.label} removed sample {sample_name}")
                self._request_rebuild()
                return

            # Extract PAM value: first element of y data
            y_data = data_val.get("y", [])
            pam_value = ""
            if len(y_data) > 0:
                try:
                    pam_value = str(y_data[0])
                except Exception:
                    pam_value = str(y_data)

            # Create a new row based on current metadata
            new_row = self.current_metadata.copy()
            new_row["_internal_uuid"] = sample_uuid

            # Replace field if selected
            if dpg.does_item_exist(self.replace_combo_tag):
                replace_field = dpg.get_value(self.replace_combo_tag)
                if replace_field and replace_field != "None" and replace_field in new_row:
                    new_row[replace_field] = sample_name

            if self.fill_with_data_key and self.fill_with_data_key in new_row:
                new_row[self.fill_with_data_key] = pam_value
            self.rows.append(new_row)
            self._request_rebuild()

    def _update_combo_choices(self, default_populate: str = "None") -> None:
        if not dpg.does_item_exist(self.replace_combo_tag):
            return
        choices = ["None"] + self.metadata_keys
        current_val = dpg.get_value(self.replace_combo_tag)
        dpg.configure_item(self.replace_combo_tag, items=choices)
        if current_val not in choices or current_val == "None":
            if default_populate in choices:
                dpg.set_value(self.replace_combo_tag, default_populate)
            else:
                dpg.set_value(self.replace_combo_tag, "None")

    def _clear_table(self, *args, **kwargs) -> None:
        self.rows = []
        self._request_rebuild()

    def _request_rebuild(self) -> None:
        """Rebuild the table safely, instantly if on the main thread, or scheduled if on a background thread."""
        if threading.current_thread() == threading.main_thread():
            self._rebuild_table()
        else:
            if self._rebuild_timer:
                self._rebuild_timer.cancel()

            def on_timer() -> None:
                dpg.set_frame_callback(dpg.get_frame_count() + 3, lambda: self._rebuild_table())

            self._rebuild_timer = threading.Timer(0.05, on_timer)
            self._rebuild_timer.start()

    def _rebuild_table(self) -> None:
        if not dpg.does_item_exist(self.table_container_tag):
            return

        if dpg.does_item_exist(self.table_tag):
            dpg.delete_item(self.table_tag)

        if not self.metadata_keys:
            table_id = dpg.add_table(parent=self.table_container_tag, tag=self.table_tag, header_row=True)
            dpg.add_table_column(parent=table_id, label="Waiting for metadata...")
            return

        table_id = dpg.add_table(
            parent=self.table_container_tag,
            tag=self.table_tag,
            header_row=True,
            resizable=True,
            borders_outerH=True,
            borders_innerV=True,
            borders_innerH=True,
            borders_outerV=True,
            scrollY=True,
            scrollX=True,
        )

        # Add columns
        dpg.add_table_column(parent=table_id, label="Actions", width_fixed=True, init_width_or_weight=40)
        for col in self.metadata_keys:
            dpg.add_table_column(parent=table_id, label=col, width_fixed=True, init_width_or_weight=100)

        is_user = app_state.mode == "user"

        # Add rows
        for r_idx, row_dict in enumerate(list(self.rows)):
            row_id = dpg.add_table_row(parent=table_id)
            dpg.add_button(
                parent=row_id,
                label="X",
                callback=self._delete_row,
                user_data=r_idx,
                small=True,
                enabled=not is_user,
            )
            for col in self.metadata_keys:
                val = row_dict.get(col, "")
                tag = f"agreg_cell_{self.UUID}_{r_idx}_{col}"
                is_editable = self.editable_fields.get(col, True)
                if is_editable:
                    is_readonly = is_user if col.lower() not in ["genotype", "genotypes", "génotype", "génotypes"] else False
                    dpg.add_input_text(
                        parent=row_id,
                        default_value=str(val),
                        tag=tag,
                        width=-1,
                        callback=self._on_cell_edited,
                        user_data=(r_idx, col),
                        readonly=is_readonly,
                    )
                else:
                    dpg.add_text(parent=row_id, default_value=str(val), tag=tag)

    def _on_cell_edited(self, sender: Any, app_data: Any, user_data: Any, *args, **kwargs) -> None:
        r_idx, col = user_data
        if 0 <= r_idx < len(self.rows):
            self.rows[r_idx][col] = app_data

    def _delete_row(self, sender: Any, app_data: Any, user_data: Any, *args, **kwargs) -> None:
        r_idx = user_data
        if 0 <= r_idx < len(self.rows):
            self.rows.pop(r_idx)
            self._request_rebuild()

    def _export_csv(self, *args: Any, **kwargs: Any) -> None:
        from core.tutorial_manager import tutorial_manager

        if not self.rows:
            tutorial_manager.play_error_hint(self.table_container_tag, "The table is empty! There are no rows to export.")
            logger.warning("Export aborted: The table is empty.")
            return

        cols = self.metadata_keys

        # Validate that all cells are filled before exporting
        for r_idx, row_dict in enumerate(self.rows):
            for col in cols:
                val = str(row_dict.get(col, "")).strip()
                if not val:
                    cell_tag = f"agreg_cell_{self.UUID}_{r_idx}_{col}"
                    tutorial_manager.play_error_hint(
                        cell_tag, f"This cell in column '{col}' is empty! Please complete it before exporting."
                    )
                    logger.warning(f"Export aborted: Cell in row {r_idx}, column '{col}' is empty.")
                    return

        from core.file_explorer import file_explorer

        def save_callback(file_path: str | None, selected_ext: str, *args, **kwargs) -> None:
            if file_path:
                if not file_path.endswith(".csv"):
                    file_path += ".csv"
                try:
                    with open(file_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=cols)
                        writer.writeheader()
                        for r in self.rows:
                            # only write known columns just in case
                            row_data = {k: r.get(k, "") for k in cols}
                            writer.writerow(row_data)
                    logger.success(f"TARA Aggregator exported to {file_path}")
                except Exception as e:
                    logger.error(f"Failed to export CSV: {e}")

        file_explorer.save_file(
            default_path=self.default_save_path,
            default_name="tara_agregator_export.csv",
            extensions=[("CSV files", "*.csv")],
            callback=save_callback,
        )


EXPORTED_CLASS = TaraAggregatorWin
EXPORTED_NAME = "TARA Agregator"
