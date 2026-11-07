from __future__ import annotations

from typing import Any

import dearpygui.dearpygui as dpg
from loguru import logger

from config.display_scaling import display_scaling
from config.theme_manager import theme_manager
from core.app_state import app_state
from core.input_output_types import IOTypes
from core.window_base import WindowBase


class RoiManagerWin(WindowBase):
    """
    ROI Manager window for selecting, naming, reordering, and managing regions of interest (ROIs).
    Coordinates ROI creation, selection state, and deletions with connected Image Viewers and Processors.
    """

    label: str = "ROI Manager"
    win_width: int = display_scaling.scale(300)
    win_height: int = display_scaling.scale(400)
    description: str = (
        "Region of Interest (ROI) Manager: create, label, filter, reorder (drag & drop), "
        "select, and manage biological ROIs across multi-channel workflows."
    )

    accepted_input_types: list[str] = [
        IOTypes.ROI_SAMPLE,
        IOTypes.CMD_DICT,
    ]
    outputs: dict[str, str] = {
        "Updates": IOTypes.ROI_SAMPLE,
        "CMD": IOTypes.CMD_DICT,
    }

    persistent_fields: list[str] = [
        "label",
        "silent_clear_on_cmd",
        "filter_text",
    ]

    # Class-level defaults for persistent fields (guarantees safe initialization)
    silent_clear_on_cmd: bool = False
    filter_text: str = ""

    def __init__(
        self,
        label: str = "ROI Manager",
        win_width: int = 300,
        win_height: int = 400,
        pos: tuple[int, int] = (10, 10),
        uuid: str | None = None,
        outputs: dict[str, Any] | None = None,
        visible: bool = True,
        silent_clear_on_cmd: bool | None = None,
        filter_text: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs["label"] = label
        kwargs["win_width"] = display_scaling.scale(win_width)
        kwargs["win_height"] = display_scaling.scale(win_height)
        kwargs["pos"] = pos
        kwargs["uuid"] = uuid
        kwargs["visible"] = visible
        if outputs is not None:
            kwargs["outputs"] = outputs
        else:
            kwargs.setdefault("outputs", dict(self.outputs))

        super().__init__(**kwargs)
        self._is_closing: bool = False

        # Defensive recovery of persistent fields
        self.silent_clear_on_cmd = bool(
            getattr(self, "silent_clear_on_cmd", False)
            if silent_clear_on_cmd is None
            else silent_clear_on_cmd
        )
        self.filter_text = str(
            getattr(self, "filter_text", "")
            if filter_text is None
            else filter_text
        )
        self._filter_query: str = self.filter_text.strip().lower()

        self.current_group: str | None = None

        # UI Tags
        self.clear_rois_tag: str = f"clear_rois_{self.UUID}"
        self.rois_group_tag: str = f"rois_group_{self.UUID}"
        self.select_all_btn_tag: str = f"roi_select_all_{self.UUID}"
        self.deselect_all_btn_tag: str = f"roi_deselect_all_{self.UUID}"
        self.filter_input_tag: str = f"roi_filter_{self.UUID}"
        self.clear_filter_btn_tag: str = f"clear_roi_filter_{self.UUID}"
        self._handler_registry_tag: str = f"roi_mgr_handler_reg_{self.UUID}"

        self.rois_dict: dict[Any, dict[str, Any]] = {}  # Store ROIs by UUID (insertion order = display order)
        self._assigned_ids: set[int] = set()  # Track numeric IDs for reuse
        self.highlighted_roi: Any | None = None

        self._dragging_from: Any | None = None  # UUID of the handle being dragged
        self._was_mouse_down: bool = False  # Previous mouse-down state
        self._handle_to_uuid: dict[str, Any] = {}  # handle_tag (str) -> roi_uuid

        # Themes
        self._handle_theme = dpg.add_theme()
        with dpg.theme_component(dpg.mvButton, parent=self._handle_theme):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (60, 60, 60, 180), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (60, 180, 80, 220), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (40, 150, 60, 255), category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, display_scaling.scale(3.0))

        self._build_interface()
        self.update_permission()

    def _build_interface(self) -> None:
        """Constructs the ROI Manager user interface using display scaling."""
        s = display_scaling.scale

        with self.window():
            dpg.add_button(
                label="Clear ROIs",
                width=-1,
                tag=self.clear_rois_tag,
                callback=self.clear_rois_cb,
            )
            dpg.add_button(
                label="Select all",
                width=-1,
                tag=self.select_all_btn_tag,
                callback=self.select_all_rois_cb,
            )
            dpg.add_button(
                label="Deselect all",
                width=-1,
                tag=self.deselect_all_btn_tag,
                callback=self.deselect_all_rois_cb,
            )
            with dpg.group(horizontal=True):
                dpg.add_input_text(
                    hint="Filter by name...",
                    default_value=self.filter_text,
                    tag=self.filter_input_tag,
                    callback=self._on_filter_changed,
                    width=-s(32),
                )
                dpg.add_button(
                    label="X",
                    width=s(26),
                    tag=self.clear_filter_btn_tag,
                    callback=self._clear_filter_cb,
                )

            dpg.add_separator()
            dpg.add_group(tag=self.rois_group_tag)

        with dpg.handler_registry(tag=self._handler_registry_tag):
            dpg.add_mouse_move_handler(callback=self.on_mouse_move)

    def update_permission(self) -> None:
        """
        Adjust module permissions and UI visibility based on app_state.mode.
        Disable closing in user mode.
        """
        is_user = (app_state.mode == "user")
        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def _matches_filter(self, name: str) -> bool:
        """Check whether a ROI name matches the current filter text (case-insensitive)."""
        if not self._filter_query:
            return True
        return self._filter_query in str(name).lower()

    def _on_filter_changed(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        """Update filter string and refresh visible/selected states of all ROIs."""
        if app_data is not None:
            self.filter_text = str(app_data)
        elif dpg.does_item_exist(self.filter_input_tag):
            self.filter_text = str(dpg.get_value(self.filter_input_tag) or "")
        self._filter_query = self.filter_text.strip().lower()
        self._apply_filter()

    def _clear_filter_cb(self, *args: Any, **kwargs: Any) -> None:
        """Clear the filter input text and restore visibility of all ROIs."""
        self.filter_text = ""
        self._filter_query = ""
        if dpg.does_item_exist(self.filter_input_tag):
            dpg.set_value(self.filter_input_tag, "")
        self._apply_filter()

    def _apply_filter(self) -> None:
        """Show matching ROIs, hide non-matching ROIs, and auto-deselect hidden ones."""
        for roi_uuid, roi_data in self.rois_dict.items():
            name = roi_data.get("name", "")
            matches = self._matches_filter(name)
            roi_data["visible"] = matches
            group_tag = f"roi_group_{roi_uuid}"
            if dpg.does_item_exist(group_tag):
                dpg.configure_item(group_tag, show=matches)

            if not matches and roi_data.get("selected", False):
                self._set_selection(roi_uuid, False)
                self._send_update(roi_uuid, "unselect")

    def input_cb(self, *args: Any, **kwargs: Any) -> None:
        """Accept ROI_SAMPLE and CMD_DICT input."""
        data_type = kwargs.get("data_type")

        if data_type == IOTypes.CMD_DICT or (
            isinstance(kwargs.get("data"), dict)
            and ("rename" in kwargs["data"] or "clear" in kwargs["data"] or kwargs["data"].get("action") == "clear")
        ):
            cmd = kwargs.get("data", kwargs.get("sample", {}))
            if cmd.get("clear") or cmd.get("action") == "clear":
                visited = list(cmd.get("visited", []))
                if self.UUID not in visited:
                    cmd["visited"] = visited + [self.UUID]
                    silent = self.silent_clear_on_cmd
                    group = cmd.get("group")
                    if group is not None:
                        if group != self.current_group:
                            silent = True
                        self.current_group = group
                    self.clear_rois_cb(cmd=cmd, silent=silent)

            if "rename" in cmd and isinstance(cmd["rename"], list):
                names = cmd["rename"]
                for i, roi_uuid in enumerate(list(self.rois_dict.keys())):
                    if i < len(names):
                        new_name = str(names[i])
                        self.rois_dict[roi_uuid]["name"] = new_name
                        name_tag = f"roi_name_{roi_uuid}"
                        if dpg.does_item_exist(name_tag):
                            dpg.set_value(name_tag, new_name)
                        self._send_update(roi_uuid, "rename")
                self._apply_filter()
            return

        sample = None
        if data_type == IOTypes.ROI_SAMPLE:
            sample = kwargs.get("sample")
        elif "sample" in kwargs:
            sample = kwargs["sample"]

        if not sample:
            return

        action = sample.get("action")
        if action == "clear" or sample.get("clear"):
            visited = list(sample.get("visited", []))
            if self.UUID in visited:
                return
            sample["visited"] = visited + [self.UUID]
            self.clear_rois_cb(cmd=sample)
            return

        roi_uuid = sample.get("uuid")
        if not roi_uuid:
            return

        if action == "create":
            self._add_roi(sample)
        elif action == "delete":
            self._remove_roi(roi_uuid, silent=True)
        elif action == "select":
            roi_name = self.rois_dict.get(roi_uuid, {}).get("name", "")
            if self._matches_filter(roi_name):
                self._set_selection(roi_uuid, True)
            else:
                self._set_selection(roi_uuid, False)
                self._send_update(roi_uuid, "unselect")
        elif action == "unselect":
            self._set_selection(roi_uuid, False)

    def _add_roi(self, sample: dict[str, Any]) -> None:
        roi_uuid = sample["uuid"]
        name = sample["name"]

        # If name is numeric, track it for reuse logic
        if str(name).isdigit():
            self._assigned_ids.add(int(name))

        matches = self._matches_filter(name)
        self.rois_dict[roi_uuid] = sample
        self._append_roi_row(roi_uuid, name, selected=matches)
        self._send_update(roi_uuid, "create")
        if not matches:
            self._set_selection(roi_uuid, False)
            self._send_update(roi_uuid, "unselect")

    def _append_roi_row(self, roi_uuid: Any, name: str, selected: bool = True) -> None:
        """Add a single ROI row widget at the bottom of the list."""
        group_tag = f"roi_group_{roi_uuid}"
        if dpg.does_item_exist(group_tag):
            return

        s = display_scaling.scale
        matches = self._matches_filter(name)
        effective_selected = bool(selected and matches)
        if roi_uuid in self.rois_dict:
            self.rois_dict[roi_uuid]["selected"] = effective_selected
            self.rois_dict[roi_uuid]["visible"] = matches

        dpg.add_group(horizontal=True, tag=group_tag, parent=self.rois_group_tag, show=matches)

        # Drag handle — drag SOURCE + drop TARGET
        handle_tag = f"roi_handle_{roi_uuid}"
        btn_handle = dpg.add_button(
            label="\uf0c9",
            tag=handle_tag,
            parent=group_tag,
            drop_callback=self._on_drop_cb,
            payload_type="ROI_REORDER",
            user_data=roi_uuid,
        )
        theme_manager.apply_icon_font(btn_handle)
        dpg.add_drag_payload(
            parent=handle_tag,
            payload_type="ROI_REORDER",
            drag_data=roi_uuid,
        )
        dpg.bind_item_theme(handle_tag, self._handle_theme)
        self._handle_to_uuid[str(handle_tag)] = roi_uuid  # Register for O(1) drop lookup

        dpg.add_checkbox(
            tag=f"roi_checkbox_{roi_uuid}",
            default_value=effective_selected,
            callback=self.roi_checkbox_cb,
            user_data=roi_uuid,
            parent=group_tag,
        )

        dpg.add_input_text(
            tag=f"roi_name_{roi_uuid}",
            default_value=name,
            width=s(100),
            callback=self.roi_name_cb,
            user_data=roi_uuid,
            parent=group_tag,
        )

        btn_del = dpg.add_button(
            label="\uf2ed",
            tag=f"roi_delete_{roi_uuid}",
            callback=self.delete_roi_cb,
            user_data=roi_uuid,
            parent=group_tag,
        )
        theme_manager.apply_icon_font(btn_del)

    def _rebuild_ui(self) -> None:
        """Rebuild the entire ROI list UI in current dict order."""
        self._handle_to_uuid.clear()
        state = {}
        for roi_uuid, roi_data in list(self.rois_dict.items()):
            state[roi_uuid] = {
                "selected": roi_data.get("selected", True),
                "name": roi_data.get("name", ""),
            }
            group_tag = f"roi_group_{roi_uuid}"
            if dpg.does_item_exist(group_tag):
                dpg.delete_item(group_tag)

        for roi_uuid in list(self.rois_dict.keys()):
            s = state.get(roi_uuid, {})
            name = s.get("name", self.rois_dict[roi_uuid].get("name", ""))
            selected = s.get("selected", True)
            self.rois_dict[roi_uuid]["name"] = name
            self._append_roi_row(roi_uuid, name, selected=selected)

    def _on_drop_cb(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        """Handle drag and drop reordering: insert source before target."""
        # Resolve target UUID via dict — works with any UUID type (int, str, UUID)
        target_uuid = self._handle_to_uuid.get(str(sender))
        if target_uuid is None:
            logger.warning(f"DROP: unknown handle sender={sender!r}")
            return

        source_uuid = app_data  # drag_data passed from add_drag_payload

        if source_uuid == target_uuid:
            return

        if source_uuid not in self.rois_dict or target_uuid not in self.rois_dict:
            logger.warning(f"DROP: UUID not in rois_dict. source={source_uuid!r} target={target_uuid!r}")
            return

        keys = list(self.rois_dict.keys())
        # Insert source before target
        keys.remove(source_uuid)
        target_idx = keys.index(target_uuid)
        keys.insert(target_idx, source_uuid)

        self.rois_dict = {k: self.rois_dict[k] for k in keys}
        src_group = f"roi_group_{source_uuid}"
        dst_group = f"roi_group_{target_uuid}"
        if dpg.does_item_exist(src_group) and dpg.does_item_exist(dst_group):
            dpg.move_item(src_group, parent=self.rois_group_tag, before=dst_group)
        else:
            self._rebuild_ui()

    def _set_selection(self, roi_uuid: Any, selected: bool) -> None:
        if roi_uuid in self.rois_dict:
            self.rois_dict[roi_uuid]["selected"] = selected
        checkbox_tag = f"roi_checkbox_{roi_uuid}"
        if dpg.does_item_exist(checkbox_tag):
            dpg.set_value(checkbox_tag, selected)

    def roi_checkbox_cb(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        """Send select/unselect action."""
        roi_uuid = user_data
        if roi_uuid not in self.rois_dict:
            return

        self.rois_dict[roi_uuid]["selected"] = bool(app_data)
        action = "select" if app_data else "unselect"
        self._send_update(roi_uuid, action)

    def roi_name_cb(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        """Send rename action and re-evaluate filter."""
        roi_uuid = user_data
        if roi_uuid not in self.rois_dict:
            return

        roi_data = self.rois_dict[roi_uuid]
        roi_data["name"] = app_data
        self._send_update(roi_uuid, "rename")
        matches = self._matches_filter(str(app_data))
        roi_data["visible"] = matches
        group_tag = f"roi_group_{roi_uuid}"
        if dpg.does_item_exist(group_tag):
            dpg.configure_item(group_tag, show=matches)
        if not matches and roi_data.get("selected", False):
            self._set_selection(roi_uuid, False)
            self._send_update(roi_uuid, "unselect")

    def delete_roi_cb(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        """Delete generic logic."""
        roi_uuid = user_data
        self._remove_roi(roi_uuid)

    def _remove_roi(self, roi_uuid: Any, silent: bool = False) -> None:
        """Removes ROI internally and optionally notifies others."""
        if roi_uuid in self.rois_dict:
            if not silent:
                self._send_update(roi_uuid, "delete")

            old_name = self.rois_dict[roi_uuid].get("name")
            if str(old_name).isdigit():
                self._assigned_ids.discard(int(old_name))

            self._handle_to_uuid.pop(f"roi_handle_{roi_uuid}", None)
            group_tag = f"roi_group_{roi_uuid}"
            if dpg.does_item_exist(group_tag):
                dpg.delete_item(group_tag)
            del self.rois_dict[roi_uuid]

    def _send_update(self, roi_uuid: Any, action: str) -> None:
        """Emit updated sample to connected modules."""
        if roi_uuid not in self.rois_dict:
            return

        roi_data = self.rois_dict[roi_uuid]
        if action in ("select", "unselect"):
            roi_data["selected"] = (action == "select")

        targets = self.connections.get("Updates", [])
        if not targets:
            return

        sample = roi_data.copy()
        sample["action"] = action

        for module in targets:
            if hasattr(module, "input_cb"):
                try:
                    module.input_cb(sample=sample, data_type=IOTypes.ROI_SAMPLE)
                except Exception as e:
                    logger.error(f"RoiManager update error: {e}")

    def select_all_rois_cb(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        """Select all ROIs matching the current filter (keep non-matching ROIs deselected)."""
        for roi_uuid, roi_data in self.rois_dict.items():
            name = roi_data.get("name", "")
            if self._matches_filter(name):
                self._set_selection(roi_uuid, True)
                self._send_update(roi_uuid, "select")
            elif roi_data.get("selected", False):
                self._set_selection(roi_uuid, False)
                self._send_update(roi_uuid, "unselect")

    def deselect_all_rois_cb(
        self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any
    ) -> None:
        for roi_uuid in self.rois_dict:
            self._set_selection(roi_uuid, False)
            self._send_update(roi_uuid, "unselect")

    def clear_rois_cb(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        cmd: dict[str, Any] | None = None,
        silent: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        visited = list(cmd.get("visited", [])) if cmd else []
        if self.UUID not in visited:
            visited = visited + [self.UUID]

        for roi_uuid in list(self.rois_dict.keys()):
            self._remove_roi(roi_uuid, silent=silent)

        self._assigned_ids.clear()

        if not silent:
            for module in self.connections.get("Updates", []):
                if hasattr(module, "input_cb"):
                    try:
                        module.input_cb(sample={"clear": True, "visited": visited}, data_type=IOTypes.ROI_SAMPLE)
                    except Exception as e:
                        logger.error(f"RoiManager clear Updates error: {e}")

            for module in self.connections.get("CMD", []):
                if hasattr(module, "input_cb"):
                    try:
                        module.input_cb(
                            data={"clear": True, "reset_counter": True, "visited": visited},
                            data_type=IOTypes.CMD_DICT,
                        )
                    except Exception as e:
                        logger.error(f"RoiManager clear CMD error: {e}")

    def on_mouse_move(self, sender: Any = None, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Hover detection: highlight ROI in image viewer."""
        if not self.rois_dict or not dpg.does_item_exist(self.winID):
            return

        is_down = dpg.is_mouse_button_down(dpg.mvMouseButton_Left)
        is_dragging = is_down and (self._dragging_from is not None)

        try:
            is_win_hovered = dpg.is_item_hovered(self.winID)
        except Exception:
            is_win_hovered = False

        if not is_win_hovered and not is_dragging:
            if not is_down:
                self._dragging_from = None
            self._was_mouse_down = is_down
            if self.highlighted_roi is not None:
                self._send_update(self.highlighted_roi, "unhighlight")
                self.highlighted_roi = None
            return

        # Detect drag start
        if is_down and not self._was_mouse_down:
            self._dragging_from = None
            for roi_uuid, roi_data in self.rois_dict.items():
                if not roi_data.get("visible", True):
                    continue
                handle_tag = f"roi_handle_{roi_uuid}"
                try:
                    if dpg.is_item_hovered(handle_tag):
                        self._dragging_from = roi_uuid
                        break
                except Exception:
                    pass

        if not is_down:
            self._dragging_from = None
        self._was_mouse_down = is_down

        current_hover = None

        if is_dragging:
            current_hover = self._dragging_from
            try:
                _, my = dpg.get_mouse_pos(local=False)
                for roi_uuid, roi_data in self.rois_dict.items():
                    if roi_uuid == self._dragging_from or not roi_data.get("visible", True):
                        continue
                    group_tag = f"roi_group_{roi_uuid}"
                    rmin = dpg.get_item_rect_min(group_tag)
                    rmax = dpg.get_item_rect_max(group_tag)
                    if rmin[1] <= my <= rmax[1]:
                        current_hover = roi_uuid
                        break
            except Exception:
                pass
        else:
            for roi_uuid, roi_data in self.rois_dict.items():
                if not roi_data.get("visible", True):
                    continue
                cb_tag = f"roi_checkbox_{roi_uuid}"
                handle_tag = f"roi_handle_{roi_uuid}"
                try:
                    if dpg.is_item_hovered(cb_tag) or dpg.is_item_hovered(handle_tag):
                        current_hover = roi_uuid
                        break
                except Exception:
                    pass

        if current_hover != self.highlighted_roi:
            if self.highlighted_roi is not None:
                self._send_update(self.highlighted_roi, "unhighlight")
            if current_hover is not None:
                self._send_update(current_hover, "highlight")
            self.highlighted_roi = current_hover

    def is_ready(self) -> bool:
        """Indicate whether the module is initialized and ready to operate."""
        return not getattr(self, "_is_closing", False)

    def on_close(self) -> None:
        """Release resources, reset hover highlighting, and invoke base teardown."""
        if getattr(self, "_is_closing", False):
            return
        self._is_closing = True

        if self.highlighted_roi is not None:
            try:
                self._send_update(self.highlighted_roi, "unhighlight")
            except Exception:
                pass
            self.highlighted_roi = None

        if hasattr(self, "_handler_registry_tag") and dpg.does_item_exist(self._handler_registry_tag):
            try:
                dpg.delete_item(self._handler_registry_tag)
            except Exception:
                pass
        if hasattr(self, "_handle_theme") and dpg.does_item_exist(self._handle_theme):
            try:
                dpg.delete_item(self._handle_theme)
            except Exception:
                pass

        super().on_close()

    def close(self) -> None:
        """Close window and unregister handlers/themes."""
        self.on_close()
        super().close()

    def destroy(self) -> None:
        """Destroy window and release resources."""
        self.close()
        super().destroy()

EXPORTED_CLASS = RoiManagerWin
EXPORTED_NAME = "ROI Manager"

