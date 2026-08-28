from __future__ import annotations
import dearpygui.dearpygui as dpg
from core.window_base import WindowBase
from core.input_output_types import IOTypes
from typing import Any

class RoiManagerWin(WindowBase):
	"""
	ROI Manager window for selecting and managing regions of interest.
	"""
	_persistent_fields: list[str]
	accepted_input_types: list[IOTypes]
	outputs: dict[str, IOTypes]
	connections: dict[str, list[Any]]
	clear_rois_tag: str
	rois_group_tag: str
	select_all_btn_tag: str
	deselect_all_btn_tag: str
	rois_dict: dict[Any, dict[str, Any]]
	_assigned_ids: set[int]
	highlighted_roi: Any | None
	_dragging_from: Any | None
	_was_mouse_down: bool
	_handle_to_uuid: dict[str, Any]
	_handle_theme: int | str

	def __init__(self,
				label: str = "ROI Manager",
				win_width: int = 300,
				win_height: int = 400,
				pos: tuple[int, int] = (10, 10),
				uuid: str | None = None,
				outputs: dict[str, Any] | None = None,
				visible: bool = True,
				silent_clear_on_cmd: bool = False,
				**kwargs: Any) -> None: 

		super().__init__(label=label,pos=pos,win_width=win_width,win_height=win_height,uuid=uuid,outputs=outputs,visible=visible)

		self.outputs = outputs or {}
		self.connections = {k: [] for k in self.outputs}
		if not self.outputs:
			self.outputs = {
				"Updates": IOTypes.ROI_SAMPLE,
				"CMD": IOTypes.CMD_DICT
			}
			self.connections = {k: [] for k in self.outputs}

		self.silent_clear_on_cmd = silent_clear_on_cmd
		self.current_group = None
		self._persistent_fields = ["label", "silent_clear_on_cmd"]
		self.accepted_input_types = [IOTypes.ROI_SAMPLE, IOTypes.CMD_DICT]
		
		self.clear_rois_tag = f"clear_rois_{self.UUID}"
		self.rois_group_tag = f"rois_group_{self.UUID}"
		self.select_all_btn_tag = f"roi_select_all_{self.UUID}"
		self.deselect_all_btn_tag = f"roi_deselect_all_{self.UUID}"

		self.rois_dict = {}  # Store ROIs by UUID (insertion order = display order)
		self._assigned_ids = set() # Track numeric IDs for reuse
		self.highlighted_roi = None

		self._dragging_from = None     # UUID of the handle being dragged
		self._was_mouse_down = False   # Previous mouse-down state
		self._handle_to_uuid = {}      # handle_tag (str) → roi_uuid — supports any UUID type

		# ── Themes ─────────────────────────────────────────────────────────────
		# Handle button: green tint on hover
		self._handle_theme = dpg.add_theme()
		with dpg.theme_component(dpg.mvButton, parent=self._handle_theme):
			dpg.add_theme_color(dpg.mvThemeCol_Button,        (60,  60,  60,  180), category=dpg.mvThemeCat_Core)
			dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (60, 180,  80,  220), category=dpg.mvThemeCat_Core)
			dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  (40, 150,  60,  255), category=dpg.mvThemeCat_Core)

		with dpg.window(label=self.label,
						width=self.win_width,
						height=self.win_height,
						pos=self.pos,
						tag=self.winID,
						show=self.visible):
	
			dpg.add_button(label="Clear ROIs", width=-1, tag=self.clear_rois_tag, callback=self.clear_rois_cb)
			
			dpg.add_button(label="Select all", width=-1, tag=self.select_all_btn_tag, callback=self.select_all_rois_cb)
			dpg.add_button(label="Deselect all", width=-1, tag=self.deselect_all_btn_tag, callback=self.deselect_all_rois_cb)

			dpg.add_separator()
			dpg.add_group(tag=self.rois_group_tag)

			with dpg.handler_registry():
				dpg.add_mouse_move_handler(callback=self.on_mouse_move)

			self.update_permission()

	def update_permission(self) -> None:
		"""
		Adjust module permissions and UI visibility based on app_state.mode.
		Disable closing in user mode.
		"""
		from core.app_state import app_state
		mode = app_state.mode
		is_user = (mode == "user")
		
		if dpg.does_item_exist(self.winID):
			dpg.configure_item(self.winID, no_close=is_user)

	def input_cb(self, *args: Any, **kwargs: Any) -> None:
		"""Accept ROI_SAMPLE input."""
		data_type = kwargs.get("data_type")
		
		if data_type == IOTypes.CMD_DICT or (isinstance(kwargs.get("data"), dict) and ("rename" in kwargs["data"] or "clear" in kwargs["data"] or kwargs["data"].get("action") == "clear")):
			cmd = kwargs.get("data", kwargs.get("sample", {}))
			if cmd.get("clear") or cmd.get("action") == "clear":
				visited = cmd.get("visited", [])
				if self.UUID in visited:
					pass # Don't return, might be rename too
				else:
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
						if dpg.does_item_exist(f"roi_name_{roi_uuid}"):
							dpg.set_value(f"roi_name_{roi_uuid}", new_name)
						self._send_update(roi_uuid, "rename")
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
			visited = sample.get("visited", [])
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
			self._set_selection(roi_uuid, True)
		elif action == "unselect":
			self._set_selection(roi_uuid, False)

	def _add_roi(self, sample: dict[str, Any]) -> None:
		roi_uuid = sample["uuid"]
		name = sample["name"]
		
		# If name is numeric, track it for reuse logic
		if str(name).isdigit():
			self._assigned_ids.add(int(name))
			
		self.rois_dict[roi_uuid] = sample
		self._append_roi_row(roi_uuid, name, selected=True)
		self._send_update(roi_uuid, "create")

	def _append_roi_row(self, roi_uuid: Any, name: str, selected: bool = True) -> None:
		"""Add a single ROI row widget at the bottom of the list."""
		group_tag = f"roi_group_{roi_uuid}"
		if dpg.does_item_exist(group_tag):
			return

		dpg.add_group(horizontal=True, tag=group_tag, parent=self.rois_group_tag)

		# Drag handle — drag SOURCE + drop TARGET
		handle_tag = f"roi_handle_{roi_uuid}"
		dpg.add_button(
			label="\uf0c9",
			tag=handle_tag,
			parent=group_tag,
			drop_callback=self._on_drop_cb,
			payload_type="ROI_REORDER",
			user_data=roi_uuid,
		)
		dpg.add_drag_payload(
			parent=handle_tag,
			payload_type="ROI_REORDER",
			drag_data=roi_uuid,
		)
		dpg.bind_item_theme(handle_tag, self._handle_theme)
		self._handle_to_uuid[handle_tag] = roi_uuid  # Register for O(1) drop lookup

		dpg.add_checkbox(
			tag=f"roi_checkbox_{roi_uuid}", 
			default_value=selected,
			callback=self.roi_checkbox_cb,
			user_data=roi_uuid,
			parent=group_tag,
		)
		
		dpg.add_input_text(
			tag=f"roi_name_{roi_uuid}", 
			default_value=name, 
			width=100,
			callback=self.roi_name_cb,
			user_data=roi_uuid,
			parent=group_tag,
		)

		dpg.add_button(label="\uf2ed", tag=f"roi_delete_{roi_uuid}", callback=self.delete_roi_cb, user_data=roi_uuid, parent=group_tag)

		self.update_permission()

	def _rebuild_ui(self) -> None:
		"""Rebuild the entire ROI list UI in current dict order."""
		self._handle_to_uuid.clear()
		state = {}
		for roi_uuid in list(self.rois_dict.keys()):
			cb_tag = f"roi_checkbox_{roi_uuid}"
			name_tag = f"roi_name_{roi_uuid}"
			state[roi_uuid] = {
				"selected": dpg.get_value(cb_tag) if dpg.does_item_exist(cb_tag) else True,
				"name": dpg.get_value(name_tag) if dpg.does_item_exist(name_tag) else self.rois_dict[roi_uuid].get("name", "")
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

	def _on_drop_cb(self, sender: int | str, app_data: Any, user_data: Any) -> None:
		"""Handle drag and drop reordering: insert source before target."""
		from loguru import logger

		# Resolve target UUID via dict — works with any UUID type (int, str, UUID)
		target_uuid = self._handle_to_uuid.get(str(sender))
		if target_uuid is None:
			logger.warning(f"DROP: unknown handle sender={sender!r}")
			return

		source_uuid = app_data  # drag_data passed from add_drag_payload

		if source_uuid == target_uuid:
			return

		keys = list(self.rois_dict.keys())
		if source_uuid not in keys or target_uuid not in keys:
			logger.warning(f"DROP: UUID not in rois_dict. source={source_uuid!r} target={target_uuid!r}")
			return

		# Insert source before target
		keys.remove(source_uuid)
		target_idx = keys.index(target_uuid)
		keys.insert(target_idx, source_uuid)

		self.rois_dict = {k: self.rois_dict[k] for k in keys}
		self._rebuild_ui()

	def _set_selection(self, roi_uuid: Any, selected: bool) -> None:
		checkbox_tag = f"roi_checkbox_{roi_uuid}"
		if dpg.does_item_exist(checkbox_tag):
			dpg.set_value(checkbox_tag, selected)

	def roi_checkbox_cb(self, sender: int | str, app_data: Any, user_data: Any, *args: Any) -> None:
		"""Send select/unselect action."""
		roi_uuid = user_data
		if roi_uuid not in self.rois_dict: return

		action = "select" if app_data else "unselect"
		self._send_update(roi_uuid, action)

	def roi_name_cb(self, sender: int | str, app_data: Any, user_data: Any, *args: Any) -> None:
		"""Send rename action."""
		roi_uuid = user_data
		if roi_uuid not in self.rois_dict: return
		
		self.rois_dict[roi_uuid]["name"] = app_data
		self._send_update(roi_uuid, "rename")

	def delete_roi_cb(self, sender: int | str, app_data: Any, user_data: Any, *args: Any) -> None:
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
				val = int(old_name)
				if val in self._assigned_ids:
					self._assigned_ids.remove(val)

			dpg.delete_item(f"roi_group_{roi_uuid}")
			del self.rois_dict[roi_uuid]

	def _send_update(self, roi_uuid: Any, action: str) -> None:
		"""Emit updated sample."""
		if roi_uuid not in self.rois_dict: return
		
		sample = self.rois_dict[roi_uuid].copy()
		sample["action"] = action
			
		if dpg.does_item_exist(f"roi_name_{roi_uuid}"):
			sample["name"] = dpg.get_value(f"roi_name_{roi_uuid}")

		for module in self.connections.get("Updates", []):
			module.input_cb(sample=sample, data_type=IOTypes.ROI_SAMPLE)

	def select_all_rois_cb(self, sender: int | str, app_data: Any, user_data: Any = None, *args: Any) -> None:
		for roi_uuid in list(self.rois_dict.keys()):
			self._set_selection(roi_uuid, True)
			self._send_update(roi_uuid, "select")

	def deselect_all_rois_cb(self, sender: int | str, app_data: Any, user_data: Any = None, *args: Any) -> None:
		for roi_uuid in list(self.rois_dict.keys()):
			self._set_selection(roi_uuid, False)
			self._send_update(roi_uuid, "unselect")

	def clear_rois_cb(self, sender: int | str | None = None, app_data: Any = None, user_data: Any = None, cmd: dict[str, Any] | None = None, silent: bool = False, *args: Any) -> None:
		visited = cmd.get("visited", []) if cmd else []
		if self.UUID not in visited:
			visited = visited + [self.UUID]

		for roi_uuid in list(self.rois_dict.keys()):
			self._remove_roi(roi_uuid, silent=silent)
		
		self._assigned_ids.clear()

		if not silent:
			for module in self.connections.get("Updates", []):
				module.input_cb(sample={"clear": True, "visited": visited}, data_type=IOTypes.ROI_SAMPLE)
			
			for module in self.connections.get("CMD", []):
				module.input_cb(data={"clear": True, "reset_counter": True, "visited": visited}, data_type=IOTypes.CMD_DICT)

	def on_mouse_move(self, sender: int | str, data: Any, *args: Any) -> None:
		"""Hover detection: highlight ROI in image viewer."""
		is_down = dpg.is_mouse_button_down(dpg.mvMouseButton_Left)
		is_dragging = is_down and (self._dragging_from is not None)

		# ── Detect drag start ──────────────────────────────────────────────────
		if is_down and not self._was_mouse_down:
			self._dragging_from = None
			for roi_uuid in list(self.rois_dict.keys()):
				handle_tag = f"roi_handle_{roi_uuid}"
				try:
					if dpg.does_item_exist(handle_tag) and dpg.is_item_hovered(handle_tag):
						self._dragging_from = roi_uuid
						break
				except Exception:
					pass

		if not is_down:
			self._dragging_from = None
		self._was_mouse_down = is_down

		current_hover = None

		if is_dragging:
			# Default to the source ROI while dragging
			current_hover = self._dragging_from
			_, my = dpg.get_mouse_pos(local=False)
			for roi_uuid in list(self.rois_dict.keys()):
				if roi_uuid == self._dragging_from:
					continue
				group_tag = f"roi_group_{roi_uuid}"
				if not dpg.does_item_exist(group_tag):
					continue
				try:
					rmin = dpg.get_item_rect_min(group_tag)
					rmax = dpg.get_item_rect_max(group_tag)
					# Check if mouse is vertically within the row bounds
					if rmin[1] <= my <= rmax[1]:
						current_hover = roi_uuid
						break
				except Exception:
					pass
		else:
			for roi_uuid in list(self.rois_dict.keys()):
				cb_tag     = f"roi_checkbox_{roi_uuid}"
				handle_tag = f"roi_handle_{roi_uuid}"
				try:
					if (dpg.does_item_exist(cb_tag)    and dpg.is_item_hovered(cb_tag)) or \
					   (dpg.does_item_exist(handle_tag) and dpg.is_item_hovered(handle_tag)):
						current_hover = roi_uuid
						break
				except Exception:
					pass

		if current_hover != self.highlighted_roi:
			if self.highlighted_roi:
				self._send_update(self.highlighted_roi, "unhighlight")
			if current_hover:
				self._send_update(current_hover, "highlight")
			self.highlighted_roi = current_hover

EXPORTED_CLASS = RoiManagerWin
EXPORTED_NAME = "ROI Manager"
