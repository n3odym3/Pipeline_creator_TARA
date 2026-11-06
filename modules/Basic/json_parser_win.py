from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import dearpygui.dearpygui as dpg
from loguru import logger

from config.display_scaling import display_scaling
from core.input_output_types import IOTypes
from core.window_base import WindowBase


class JSONParser_win(WindowBase):
    """
    Parses JSON text into a CMD_DICT for downstream nodes.
    """

    label: str = "JSON Parser"
    win_width: int = display_scaling.scale(260)
    win_height: int = display_scaling.scale(200)
    description: str = "Parse incoming JSON strings into structured dictionaries (CMD_DICT) downstream."

    accepted_input_types: list[str] = [IOTypes.TEXT]
    outputs: dict[str, str] = {
        "Dict": IOTypes.CMD_DICT,
    }
    persistent_fields: list[str] = ["label"]

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the JSON Parser window module.

        Args:
            **kwargs: Additional parameters forwarded to WindowBase.
        """
        super().__init__(**kwargs)

        self.status_tag: str = f"json_status_{self.UUID}"
        self.preview_tag: str = f"json_preview_{self.UUID}"

        self._build_ui()
        self.update_permission()

    def _build_ui(self) -> None:
        """Construct DearPyGui widgets for JSON Parser."""
        with self.window():
            dpg.add_text("Ready", tag=self.status_tag)
            dpg.add_separator()
            dpg.add_input_text(
                tag=self.preview_tag,
                multiline=True,
                readonly=True,
                width=-1,
                height=-1,
            )

    def update_permission(self) -> None:
        """
        Adjust module permissions and UI elements based on the application mode.
        """
        from core.app_state import app_state

        is_user = getattr(app_state, "mode", "dev") == "user"
        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def input_cb(self, *args: Any, **kwargs: Any) -> None:
        """
        Receive input JSON string, parse it, update the UI and send the resulting dict downstream.
        """
        data = kwargs.get("data") if "data" in kwargs else (args[0] if len(args) > 0 else None)
        if data is None or not isinstance(data, str):
            return

        try:
            parsed = json.loads(data)

            if isinstance(parsed, dict):
                if dpg.does_item_exist(self.status_tag):
                    dpg.set_value(self.status_tag, "Status: OK (Dict)")
                    dpg.configure_item(self.status_tag, color=(0, 255, 0, 255))

                if dpg.does_item_exist(self.preview_tag):
                    preview_text = json.dumps(parsed, indent=2)
                    dpg.set_value(self.preview_tag, preview_text)

                self.trigger_cb(parsed)
            else:
                if dpg.does_item_exist(self.status_tag):
                    dpg.set_value(self.status_tag, "Status: Valid JSON, but not a dict")
                    dpg.configure_item(self.status_tag, color=(255, 165, 0, 255))

                if dpg.does_item_exist(self.preview_tag):
                    dpg.set_value(self.preview_tag, str(parsed))
        except json.JSONDecodeError as e:
            if dpg.does_item_exist(self.status_tag):
                dpg.set_value(self.status_tag, "Status: JSON Error")
                dpg.configure_item(self.status_tag, color=(255, 0, 0, 255))

            if dpg.does_item_exist(self.preview_tag):
                dpg.set_value(self.preview_tag, str(e))
            logger.error(f"[{self.label}] JSON Decode Error: {e}")

    def trigger_cb(
        self,
        cmd_dict: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Send the parsed dictionary to downstream modules.
        """
        for output_key in self.outputs:
            for module in self.connections.get(output_key, []):
                try:
                    module.input_cb(cmd_dict, data=cmd_dict, data_type=IOTypes.CMD_DICT)
                except Exception as e:
                    logger.error(
                        f"[{self.label}] Error sending to {getattr(module, 'label', 'module')}: {e}"
                    )


EXPORTED_CLASS = JSONParser_win
EXPORTED_NAME = "JSON Parser"
