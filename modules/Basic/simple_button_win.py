from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import dearpygui.dearpygui as dpg
from loguru import logger

from config.display_scaling import display_scaling
from core.input_output_types import IOTypes
from core.window_base import WindowBase


class Button_win(WindowBase):
    """
    A simple DearPyGui window containing a button that acts as a trigger source.
    """

    label: str = "Button"
    win_width: int = display_scaling.scale(180)
    win_height: int = display_scaling.scale(90)
    description: str = "Manual trigger button emitting a TRIGGER signal to downstream connected modules."

    accepted_input_types: list[str] = [IOTypes.TRIGGER]
    outputs: dict[str, str] = {
        "Trig": IOTypes.TRIGGER,
    }
    persistent_fields: list[str] = ["label"]

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the Trigger Button window module.

        Args:
            **kwargs: Additional parameters forwarded to WindowBase.
        """
        super().__init__(**kwargs)

        self.trigger_btn_tag: str = f"button_trig_{self.UUID}"

        self._build_ui()
        self.update_permission()

    def _build_ui(self) -> None:
        """Construct DearPyGui widgets for the button module."""
        with self.window():
            dpg.add_button(
                label="Trigger",
                tag=self.trigger_btn_tag,
                callback=self.trigger_cb,
                width=-1,
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
        Receive programmatic trigger and forward it to outputs.
        """
        self.trigger_cb(*args, **kwargs)

    def trigger_cb(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Forward the trigger to all connected output modules.
        """
        for output_key, targets in self.connections.items():
            for module in targets:
                try:
                    module.input_cb(data_type=IOTypes.TRIGGER)
                except Exception as e:
                    logger.error(
                        f"[{self.label}] Error triggering {getattr(module, 'label', 'module')}: {e}"
                    )


EXPORTED_CLASS = Button_win
EXPORTED_NAME = "Trigger Button"
