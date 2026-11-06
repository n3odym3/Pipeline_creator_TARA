from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import dearpygui.dearpygui as dpg
from loguru import logger

from core.input_output_types import IOTypes
from core.window_base import WindowBase


class Gate_win(WindowBase):
    """
    Gate module window for controlling data pass-through in a pipeline.

    Behaves as a conditional relay:
    - When open (is_open=True), incoming wired data is forwarded downstream to all connected modules.
    - When closed (is_open=False), incoming transmission is blocked.
    Provides an interactive UI window with a checkbox to toggle pass-through directly.
    """

    def __init__(
        self,
        label: str = "Gate",
        win_width: int = 240,
        win_height: int = 120,
        pos: Tuple[int, int] = (10, 10),
        uuid: Optional[str] = None,
        outputs: Optional[Dict[str, Any]] = None,
        visible: bool = True,
        is_open: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a Gate window instance.

        Args:
            label: Window title and node label.
            win_width: Initial window width.
            win_height: Initial window height.
            pos: Initial window position (x, y).
            uuid: Optional unique identifier.
            outputs: Optional output dictionary.
            visible: Whether the window is visible on creation.
            is_open: Initial pass-through state (True = passing, False = blocking).
            **kwargs: Additional persistent keyword arguments.
        """
        super().__init__(
            label=label,
            pos=pos,
            win_width=win_width,
            win_height=win_height,
            uuid=uuid,
            outputs=outputs,
            visible=visible,
            **kwargs,
        )

        # Persistent fields preserved across workspace save and load
        self.persistent_fields = ["label", "is_open"]

        if "is_open" in kwargs:
            self.is_open = bool(kwargs["is_open"])
        else:
            self.is_open = bool(is_open)

        # Dynamic IOType configuration (ANY accepts all types)
        self.io_type: IOTypes = IOTypes.ANY
        self.accepted_input_types: List[IOTypes] = [self.io_type]
        self.outputs: Dict[str, IOTypes] = {"Out": self.io_type}
        self.connections: Dict[str, List[Any]] = {"Out": []}

        # DPG tag definitions
        self.checkbox_tag: str = f"gate_checkbox_{self.UUID}"
        self.status_tag: str = f"gate_status_{self.UUID}"
        self.type_tag: str = f"gate_type_{self.UUID}"

        # Build DearPyGui Window
        with dpg.window(
            label=self.label,
            width=self.win_width,
            height=self.win_height,
            pos=self.pos,
            tag=self.winID,
            show=self.visible,
        ):
            dpg.add_text("Pass-Through Control:")
            dpg.add_checkbox(
                label="Pass-through",
                tag=self.checkbox_tag,
                default_value=self.is_open,
                callback=self._on_checkbox_toggle,
            )

            status_str, status_color = self._get_status_display()
            dpg.add_text(status_str, tag=self.status_tag, color=status_color)

            dpg.add_separator()
            dpg.add_text(f"IOType: {self.get_type_str()}", tag=self.type_tag, color=(150, 200, 255))

        self.update_permission()

    def _get_status_display(self) -> Tuple[str, Tuple[int, int, int]]:
        """Return status text and color based on current pass-through state."""
        if self.is_open:
            return "Status: OPEN (Passing)", (100, 255, 100)
        return "Status: CLOSED (Blocking)", (255, 110, 110)

    def _on_checkbox_toggle(self, sender: Any, app_data: bool, user_data: Any = None, *args, **kwargs) -> None:
        """Checkbox callback in the UI window."""
        self.set_open(app_data)

    def update_permission(self) -> None:
        """
        Adjust UI permissions based on application mode:
        In user mode, prevent closing the window while keeping the pass-through checkbox accessible.
        """
        try:
            from core.app_state import app_state
            mode = app_state.mode
            is_user = mode == "user"

            if dpg.does_item_exist(self.winID):
                dpg.configure_item(self.winID, no_close=is_user)
        except Exception as e:
            logger.debug(f"Gate {self.label}: update_permission error - {e}")

    def get_type_str(self) -> str:
        """Get formatted string for current IOType."""
        return str(self.io_type.value) if hasattr(self.io_type, "value") else str(self.io_type)

    def input_cb(self, *args: Any, **kwargs: Any) -> None:
        """
        Data input callback:
        If pass-through is active (is_open=True), forward to all downstream connected modules.
        Otherwise, drop the incoming data.
        """
        if not self.is_open:
            return

        for module in self.connections.get("Out", []):
            try:
                module.input_cb(*args, **kwargs)
            except Exception as e:
                logger.error(f"Gate '{self.label}': error dispatching to {module} - {e}")

    def set_open(self, is_open: bool, *args: Any, **kwargs: Any) -> None:
        """
        Set pass-through state and synchronize UI elements.
        """
        self.is_open = bool(is_open)

        # Synchronize UI checkbox
        if dpg.does_item_exist(self.checkbox_tag):
            try:
                if dpg.get_value(self.checkbox_tag) != self.is_open:
                    dpg.set_value(self.checkbox_tag, self.is_open)
            except Exception:
                pass

        # Synchronize status text
        if dpg.does_item_exist(self.status_tag):
            try:
                status_str, status_color = self._get_status_display()
                dpg.set_value(self.status_tag, status_str)
                dpg.configure_item(self.status_tag, color=status_color)
            except Exception:
                pass

        # Refresh Node Editor wire highlights if editor exists
        try:
            from core.main_win import main_win
            if hasattr(main_win, "node_editor") and main_win.node_editor:
                main_win.node_editor.recolor_all_nodes()
        except Exception:
            pass

    def set_label(self, label: str, *args: Any, **kwargs: Any) -> None:
        """Set user-facing label of the gate."""
        clean_label = label.strip() if label else "Gate"
        self.label = clean_label

        if dpg.does_item_exist(self.winID):
            try:
                dpg.configure_item(self.winID, label=self.label)
            except Exception:
                pass

    def set_io_type(self, new_type: IOTypes) -> None:
        """Update inferred IOType and UI display."""
        if not isinstance(new_type, IOTypes):
            return

        self.io_type = new_type
        self.accepted_input_types = [new_type]
        self.outputs["Out"] = new_type

        type_str = self.get_type_str()
        if dpg.does_item_exist(self.type_tag):
            try:
                dpg.set_value(self.type_tag, f"IOType: {type_str}")
            except Exception:
                pass


EXPORTED_CLASS = Gate_win
EXPORTED_NAME = "Gate"
