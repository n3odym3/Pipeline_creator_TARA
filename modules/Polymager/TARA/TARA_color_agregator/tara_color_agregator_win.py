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


class TaraColorAggregatorWin(WindowBase):
    """
    TARA Color Aggregator Module.

    Receives:
      - Metadata from TARA (via CMD_DICT with 'metadata' key)           → defines row header columns
      - Color distribution from ColorMapper (via CMD_DICT with 'action': 'plot_pie')
          ‣ Normal emit  → updates live preview only (no row added)
          ‣ With 'save': True flag → commits a new row to the table

    The distribution dict has the shape:
        { "ColorName": { "count": int, "percent": float, "color": [R,G,B] }, … }

    Color category names become the data columns.  A 'replace' combo lets the
    user pick which metadata field is overridden by the sample name/ID supplied
    inside the distribution payload.
    """

    # Instance variables
    accepted_input_types: list[IOTypes]
    outputs: dict[str, IOTypes]
    connections: dict[str, list]
    _persistent_fields: list[str]
    metadata_keys: list[str]
    color_keys: list[str]
    editable_fields: dict[str, bool]
    fill_with_data_key: str
    optional_fields: list[str]
    rows: list[dict[str, Any]]
    default_save_path: str
    _pending_distribution: dict[str, Any] | None
    _rebuild_timer: threading.Timer | None
    table_container_tag: str
    table_tag: str
    replace_combo_tag: str
    preview_text_tag: str
    clear_btn_tag: str

    def __init__(
        self,
        label: str = "TARA Color Aggregator",
        win_width: int = 700,
        win_height: int = 420,
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

        self.accepted_input_types = [IOTypes.CMD_DICT, IOTypes.FOLDER_PATH]
        self.outputs = {}  # no outputs
        self.connections = {k: [] for k in self.outputs}
        self._persistent_fields = ["label"]

        # --- state ---
        self.metadata_keys = []  # ordered list of metadata field names
        self.color_keys = []  # ordered list of colour category names (palette)
        self.editable_fields = {}
        self.fill_with_data_key = "Pam_value"
        self.optional_fields = []
        self.rows = []  # committed rows list[dict]
        self.default_save_path = ""

        # --- live preview (not yet saved) ---
        self._pending_distribution = None
        self._rebuild_timer = None

        # --- UI tags ---
        self.table_container_tag = f"cag_table_cont_{self.UUID}"
        self.table_tag = f"cag_table_{self.UUID}"
        self.replace_combo_tag = f"cag_replace_combo_{self.UUID}"
        self.preview_text_tag = f"cag_preview_{self.UUID}"
        self.clear_btn_tag = f"cag_clear_{self.UUID}"

        # --- Build UI ---
        with dpg.window(
            label=self.label,
            width=self.win_width,
            height=self.win_height,
            pos=self.pos,
            tag=self.winID,
            show=self.visible,
        ):
            with dpg.group(horizontal=True):
                dpg.add_text("Replace field with sample name:")
                dpg.add_combo(["None"], default_value="None", tag=self.replace_combo_tag, width=180)
                dpg.add_button(label="Clear Table", tag=self.clear_btn_tag, callback=self._clear_table)
                dpg.add_button(label="Export CSV", callback=self._export_csv)

            dpg.add_text("Waiting for data…", tag=self.preview_text_tag, color=(150, 150, 150))
            dpg.add_separator()

            with dpg.child_window(tag=self.table_container_tag, border=False):
                pass

        self._rebuild_table()
        self.update_permission()

    def update_permission(self) -> None:
        """
        Adjust module permissions and UI visibility based on app_state.mode.
        Disable closing in user mode.
        """
        is_user = app_state.mode == "user"

        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

        self._request_rebuild()

    def input_cb(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
        data_val = kwargs.get("data", data)
        data_type = kwargs.get("data_type", None)

        # --- Folder path ---
        if data_type == IOTypes.FOLDER_PATH or (isinstance(data_val, str) and os.path.isdir(data_val)):
            self.default_save_path = data_val
            logger.info(f"{self.label} set save path: {self.default_save_path}")
            return

        if not isinstance(data_val, dict):
            return

        # --- Metadata from TARA ---
        if "metadata" in data_val:
            meta = data_val["metadata"]

            self.current_metadata = meta
            new_keys = list(meta.keys())

            self.editable_fields = data_val.get("editable", {})
            self.fill_with_data_key = data_val.get("fill_with_data_key", "Pam_value")
            self.optional_fields = data_val.get("optional_fields", [])
            populate_default = data_val.get("populate_default", "None")

            if new_keys != self.metadata_keys:
                self.metadata_keys = new_keys
                self._update_combo_choices(default_populate=populate_default)
                self._request_rebuild()
            return

        # --- Clear signal ---
        if data_val.get("clear", False):
            self._clear_table()
            return

        # --- Distribution from ColorMapper ---
        if data_val.get("action") == "plot_pie":
            dist = data_val.get("distribution", {})
            name = data_val.get("sample_name", "")
            save = data_val.get("save", False)

            if not dist:
                return

            # Update live colour-key list from palette (keeps order stable)
            new_color_keys = [k for k in dist.keys() if k != "best_match"]
            if new_color_keys != self.color_keys:
                self.color_keys = new_color_keys
                self._request_rebuild()

            # Update preview label
            best_match = dist.get("best_match", "?")
            pct_parts = [f"{k}: {v.get('percent', 0.0):.1f}%" for k, v in dist.items() if k != "best_match"]
            preview = f"Live → {name or '?'} | Best: {best_match} | " + "  ".join(pct_parts)
            if dpg.does_item_exist(self.preview_text_tag):
                dpg.set_value(self.preview_text_tag, preview)
                dpg.configure_item(
                    self.preview_text_tag, color=(100, 220, 255) if save else (150, 150, 150)
                )

            if not save:
                return  # preview only – do NOT add a row

            # --- Commit a row ---
            if not self.metadata_keys:
                msg = f"{self.label}: tried to save a row, but no metadata template has been set yet."
                logger.warning(msg)
                if dpg.does_item_exist(self.winID) and dpg.is_item_shown(self.winID):
                    from core.tutorial_manager import tutorial_manager
                    tutorial_manager.play_center_hint(msg)
                return

            new_row = self.current_metadata.copy()

            # Replace chosen metadata field with the sample name
            replace_field = ""
            if dpg.does_item_exist(self.replace_combo_tag):
                replace_field = dpg.get_value(self.replace_combo_tag)

            if name and replace_field and replace_field != "None" and replace_field in new_row:
                new_row[replace_field] = name
            elif name:
                new_row["_sample_name"] = name

            if self.fill_with_data_key and self.fill_with_data_key in new_row:
                new_row[self.fill_with_data_key] = best_match

            # Add each colour percentage as its own column
            for cat_name, cat_data in dist.items():
                if cat_name == "best_match":
                    new_row["Best_Match"] = cat_data
                    continue
                new_row[cat_name + "_pct"] = f"{cat_data.get('percent', 0.0):.2f}"
                new_row[cat_name + "_count"] = str(cat_data.get("count", 0))

            new_row["_internal_uuid"] = f"{name}_{len(self.rows)}"
            self.rows.append(new_row)
            logger.success(f"{self.label} added row for sample '{name}'")
            self._request_rebuild()

    def _update_combo_choices(self, default_populate: str = "None") -> None:
        if not dpg.does_item_exist(self.replace_combo_tag):
            return
        choices = ["None"] + self.metadata_keys
        current = dpg.get_value(self.replace_combo_tag)
        dpg.configure_item(self.replace_combo_tag, items=choices)
        if current not in choices or current == "None":
            if default_populate in choices:
                dpg.set_value(self.replace_combo_tag, default_populate)
            else:
                dpg.set_value(self.replace_combo_tag, "None")

    def _clear_table(self, *args: Any, **kwargs: Any) -> None:
        self.rows = []
        self._request_rebuild()

    def _request_rebuild(self) -> None:
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

        # Build full column list: metadata + Best_Match + (cat_pct, cat_count) per colour
        meta_cols = self.metadata_keys
        color_cols = ["Best_Match"]
        for k in self.color_keys:
            color_cols.append(k + "_pct")
            color_cols.append(k + "_count")

        all_cols = meta_cols + color_cols

        if not all_cols:
            table_id = dpg.add_table(parent=self.table_container_tag, tag=self.table_tag, header_row=True)
            dpg.add_table_column(parent=table_id, label="Waiting for metadata & colour data…")
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

        dpg.add_table_column(parent=table_id, label="Actions", width_fixed=True, init_width_or_weight=40)
        for col in all_cols:
            dpg.add_table_column(parent=table_id, label=col, width_fixed=True, init_width_or_weight=100)

        is_user = app_state.mode == "user"

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
            for col in all_cols:
                val = row_dict.get(col, "")
                tag = f"cag_cell_{self.UUID}_{r_idx}_{col}"

                # Check editability: colour data / best match are read-only, metadata respects config
                is_editable = True
                if col in color_cols:
                    is_editable = False
                elif col in self.editable_fields:
                    is_editable = self.editable_fields[col]

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

    def _on_cell_edited(self, sender: Any, app_data: Any, user_data: Any) -> None:
        r_idx, col = user_data
        if 0 <= r_idx < len(self.rows):
            self.rows[r_idx][col] = app_data

    def _delete_row(self, sender: Any, app_data: Any, user_data: Any) -> None:
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

        meta_cols = self.metadata_keys

        # Validate that all cells in metadata columns are filled before exporting
        for r_idx, row_dict in enumerate(self.rows):
            for col in meta_cols:
                val = str(row_dict.get(col, "")).strip()
                if not val:
                    cell_tag = f"cag_cell_{self.UUID}_{r_idx}_{col}"
                    tutorial_manager.play_error_hint(
                        cell_tag, f"This cell in column '{col}' is empty! Please complete it before exporting."
                    )
                    logger.warning(f"Export aborted: Cell in row {r_idx}, column '{col}' is empty.")
                    return

        from core.file_explorer import file_explorer

        def save_callback(file_path: str | None, selected_ext: str) -> None:
            if not file_path:
                return
            if not file_path.endswith(".csv"):
                file_path += ".csv"
            try:
                color_cols = ["Best_Match"]
                for k in self.color_keys:
                    color_cols.append(k + "_pct")
                    color_cols.append(k + "_count")
                all_cols = meta_cols + color_cols

                with open(file_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
                    writer.writeheader()
                    for r in self.rows:
                        writer.writerow({k: r.get(k, "") for k in all_cols})

                logger.success(f"TARA Color Aggregator exported to {file_path}")
                tutorial_manager.play_success_hint(self.winID, "Exported successfully!")
            except Exception as e:
                logger.error(f"Failed to export CSV: {e}")

        file_explorer.save_file(
            default_path=self.default_save_path,
            default_name="tara_color_agregator_export.csv",
            extensions=[("CSV files", "*.csv")],
            callback=save_callback,
        )


EXPORTED_CLASS = TaraColorAggregatorWin
EXPORTED_NAME = "TARA Color Agregator"
