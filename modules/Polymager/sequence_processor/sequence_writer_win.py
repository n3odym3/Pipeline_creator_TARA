from __future__ import annotations
import dearpygui.dearpygui as dpg
from loguru import logger
from core.window_base import WindowBase
from core.input_output_types import IOTypes
from .sequence_processor import sequence_processor
from .command_registry import command_registry
import os
import json
from typing import Any

class SequenceWriterWin(WindowBase):
	"""
	Sequence Writer window for composing and validating command sequences.
	"""
	_persistent_fields: list[str]
	accepted_input_types: list[IOTypes]
	outputs: dict[str, IOTypes]
	connections: dict[str, list[Any]]
	validate_seq_tag: str
	save_btn_tag: str
	registry_btn_tag: str
	help_btn_tag: str
	sequence_input_tag: str
	handler_registry_tag: str
	current_folder: str | None
	raw_sequence: str

	def __init__(self,
				label: str = "Sequence writer",
				win_width: int = 300,
				win_height: int = 200,
				pos: tuple[int, int] = (10, 10),
				uuid: str | None = None,
				outputs: dict[str, Any] | None = None,
				visible: bool = True) -> None: 

		super().__init__(label=label,pos=pos,win_width=win_width,win_height=win_height,uuid=uuid,outputs=outputs,visible=visible)

		self.outputs = outputs or {}
		self.connections = {k: [] for k in self.outputs}
		if not self.outputs:
			self.outputs = {
				"Sequence": IOTypes.TEXT,
			}
			self.connections = {k: [] for k in self.outputs}

		self._persistent_fields = ["label"]
		self.accepted_input_types = [IOTypes.TEXT, IOTypes.FOLDER_PATH, IOTypes.CMD_DICT]
		
		self.validate_seq_tag = f"seq_writer_validate_{self.UUID}"
		self.save_btn_tag = f"seq_writer_save_{self.UUID}"
		self.registry_btn_tag = f"seq_writer_registry_btn_{self.UUID}"
		self.help_btn_tag = f"seq_writer_help_btn_{self.UUID}"
		self.sequence_input_tag = f"seq_writer_input_{self.UUID}"
		self.current_folder = None

		self.raw_sequence = ""

		with dpg.window(label=self.label,
						width=self.win_width,
						height=self.win_height,
						pos=self.pos,
						tag=self.winID,
						show=self.visible):

			with dpg.group(horizontal=True):
				dpg.add_button(label="Validate", tag=self.validate_seq_tag, callback=self._validate_sequence_cb)
				dpg.add_button(label="Seq registry", tag=self.registry_btn_tag, callback=self._show_registry_cb)
				dpg.add_button(label="Save", tag=self.save_btn_tag, callback=self._on_save_clicked)
				dpg.add_button(label="Help", tag=self.help_btn_tag, callback=self._show_help_cb)

			dpg.add_input_text(
				tag=self.sequence_input_tag, 
				width=-1, height=300, multiline=True, 
				on_enter=True, 
				callback=self._validate_sequence_cb,
				payload_type="sequence", 
				drop_callback=lambda s,a :dpg.set_value(s, dpg.get_value(self.sequence_input_tag)+a))

		self.handler_registry_tag = f"seq_writer_handler_reg_{self.UUID}"
		with dpg.item_handler_registry(tag=self.handler_registry_tag):
			dpg.add_item_resize_handler(callback=self._on_resize)
		dpg.bind_item_handler_registry(self.winID, self.handler_registry_tag)

		self.update_permission()

	def _on_resize(self, *args: Any) -> None:
		if not dpg.does_item_exist(self.winID) or not dpg.does_item_exist(self.sequence_input_tag):
			return
		win_h = dpg.get_item_height(self.winID)
		if win_h and win_h > 0:
			new_h = max(40, win_h - 60)
			dpg.configure_item(self.sequence_input_tag, height=new_h)

	def update_permission(self) -> None:
		"""
		Adjust module permissions and UI visibility based on app_state.mode.
		Disable closing and lock all controls in user mode.
		"""
		from core.app_state import app_state
		mode = app_state.mode
		is_user = (mode == "user")
		
		if dpg.does_item_exist(self.winID):
			dpg.configure_item(self.winID, no_close=is_user)

		controls = [
			self.validate_seq_tag,
			self.save_btn_tag,
			self.registry_btn_tag,
			self.help_btn_tag,
			self.sequence_input_tag,
		]
		for tag in controls:
			if dpg.does_item_exist(tag):
				dpg.configure_item(tag, enabled=not is_user)

	def is_ready(self) -> bool:
		"""Check if the window is ready to process inputs."""
		return True
	
	def is_outputs_ready(self) -> bool:
		for output in self.outputs:
			is_ready = getattr(output, "is_ready", None)
			if callable(is_ready) and not is_ready():
				return False
		return True
	
	def input_cb(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
		data_type = kwargs.get("data_type")
		data_content = kwargs.get("data") if "data" in kwargs else data

		logger.info(f"Sequence Writer {self.UUID} received input: content='{data_content}', type='{data_type}'")

		if data_type == IOTypes.CMD_DICT or isinstance(data_content, dict):
			if isinstance(data_content, dict) and data_content.get("validate"):
				self._validate_sequence_cb(None, None)
				logger.info(f"Sequence Writer {self.UUID} validated sequence via command.")
			return

		if data_type == IOTypes.FOLDER_PATH or (isinstance(data_content, str) and os.path.isdir(data_content)):
			self.current_folder = data_content
			logger.info(f"Sequence Writer {self.UUID} updated tracking folder: {self.current_folder}")
			return

		if data_type == IOTypes.TEXT or isinstance(data_content, str):
			if self.current_folder and os.path.exists(self.current_folder):
				import json
				json_path = os.path.join(self.current_folder, f"{data_content}.json")
				if os.path.exists(json_path):
					try:
						with open(json_path, 'r') as f:
							payload = json.load(f)
							if "sequence" in payload:
								seq_text = payload["sequence"]
								dpg.set_value(self.sequence_input_tag, seq_text)
								self.raw_sequence = seq_text
								logger.info(f"Reloaded sequence for group '{data_content}' from {json_path}")
								return
					except Exception as e:
						logger.error(f"Failed to reload sequence from {json_path}: {e}")
				
			if data_content and ("\n" in data_content or "," in data_content):
				self.raw_sequence = data_content
				dpg.set_value(self.sequence_input_tag, self.raw_sequence)
			else:
				logger.warning(f"Sequence Writer received '{data_content}' but no corresponding JSON found in {self.current_folder}. Sequence not updated.")

	def trigger_cb(self, *args: Any, **kwargs: Any) -> None:
		"""Sends the last known status as STATUS_DICT to all outputs."""
		for idx, output_key in enumerate(self.outputs):
			connected_modules = self.connections.get(output_key, [])
			for module in connected_modules:
				if idx == 0:
					module.input_cb(self.raw_sequence)

	def _validate_sequence_cb(self, sender: int | str | None, app_data: Any, user_data: Any = None, *args: Any) -> None:
		self.raw_sequence = dpg.get_value(self.sequence_input_tag)
		self.trigger_cb()

	def _on_save_clicked(self, *args: Any, **kwargs: Any) -> None:
		from core.file_explorer import file_explorer
		default_path = os.path.join(os.path.dirname(__file__), "sequences")
		if not os.path.exists(default_path):
			os.makedirs(default_path, exist_ok=True)
		
		file_explorer.save_file(
			default_path=default_path,
			default_name="sequence.txt",
			extensions=[("Text files", "*.txt"), ("All files", "*.*")],
			callback=self._save_sequence_to_file
		)

	def _save_sequence_to_file(self, file_path: str, ext: str) -> None:
		if not file_path:
			return
		
		# Ensure extension is .txt if not specified and Text files was chosen
		if ext == ".txt" and not file_path.lower().endswith(".txt"):
			file_path += ".txt"
		elif not file_path.lower().endswith(".txt") and not os.path.splitext(file_path)[1]:
			file_path += ".txt"
			
		try:
			seq_text = dpg.get_value(self.sequence_input_tag)
			with open(file_path, "w", encoding="utf-8") as f:
				f.write(seq_text)
			logger.success(f"Sequence saved to {file_path}")
		except Exception as e:
			logger.error(f"Failed to save sequence to {file_path}: {e}")

	def _show_help_cb(self, sender: int | str, app_data: Any, user_data: Any = None, *args: Any) -> None:
		help_win_tag = f"help_win_{self.UUID}"
		if dpg.does_item_exist(help_win_tag):
			dpg.focus_item(help_win_tag)
			return
		
		with dpg.window(label="Sequence Commands Help", width=600, height=400, pos=(self.pos[0]+50, self.pos[1]+50), tag=help_win_tag, on_close=lambda s,a: dpg.delete_item(s)):
			with dpg.table(header_row=True, resizable=True, borders_innerH=True, borders_innerV=True, borders_outerH=True, borders_outerV=True):
				dpg.add_table_column(label="Command Regex", width_stretch=True, init_width_or_weight=0.3)
				dpg.add_table_column(label="Description", width_stretch=True, init_width_or_weight=0.7)
				
				for regex, func, description in command_registry:
					with dpg.table_row():
						dpg.add_text(regex.pattern)
						dpg.add_text(description)

	def _show_registry_cb(self, sender: int | str, app_data: Any, user_data: Any = None, *args: Any) -> None:
		registry_win_tag = f"seq_registry_win_{self.UUID}"
		if dpg.does_item_exist(registry_win_tag):
			dpg.focus_item(registry_win_tag)
			return
		
		# Define the base directory for sequences
		base_dir = os.path.join(os.path.dirname(__file__), "sequences")
		if not os.path.exists(base_dir):
			os.makedirs(base_dir) # create it if it doesn't exist just in case
			
		def build_ui_recursive(current_dir: str) -> None:
			try:
				items = os.listdir(current_dir)
			except OSError:
				return
				
			dirs = [d for d in items if os.path.isdir(os.path.join(current_dir, d))]
			files = [f for f in items if os.path.isfile(os.path.join(current_dir, f)) and f.endswith('.txt')]
			
			for sorted_dir in sorted(dirs):
				dir_path = os.path.join(current_dir, sorted_dir)
				# Check if this dir or any subdir has .txt files
				has_txt = False
				for root, _, fnames in os.walk(dir_path):
					if any(fname.endswith('.txt') for fname in fnames):
						has_txt = True
						break
				if has_txt:
					with dpg.collapsing_header(label=sorted_dir):
						with dpg.group(indent=20):
							build_ui_recursive(dir_path)
						
			if files:
				with dpg.group(indent=20):
					for f in sorted(files):
						file_path = os.path.join(current_dir, f)
						button_name = os.path.splitext(f)[0]
						try:
							with open(file_path, 'r', encoding='utf-8') as file_obj:
								sequence_content = file_obj.read()
						except Exception:
							sequence_content = ""

						rel_path = os.path.relpath(file_path, base_dir).replace('\\', '_').replace('/', '_').replace('.', '_')
						btn_tag = f"seq_btn_{self.UUID}_{rel_path}"
						dpg.add_button(label=button_name, width=-1, tag=btn_tag)
						
						with dpg.drag_payload(parent=btn_tag, drag_data=sequence_content, payload_type="sequence"):
							dpg.add_text(f"Dragging: {button_name}")
							
						with dpg.tooltip(btn_tag):
							dpg.add_text(sequence_content)

		with dpg.window(label="Sequence Registry", width=400, height=400, pos=(self.pos[0]+50, self.pos[1]+50), tag=registry_win_tag, on_close=lambda s,a: dpg.delete_item(s)):
			build_ui_recursive(base_dir)

EXPORTED_CLASS = SequenceWriterWin
EXPORTED_NAME = "Sequence Writer"
