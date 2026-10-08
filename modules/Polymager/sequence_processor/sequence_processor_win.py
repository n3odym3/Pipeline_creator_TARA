from __future__ import annotations
import dearpygui.dearpygui as dpg
from core.window_base import WindowBase
from core.input_output_types import IOTypes
from core.app_state import app_state
from .sequence_processor import sequence_processor
from loguru import logger
import datetime
import os
import re
import time
import threading 
from typing import Any

class SequenceProcessorWin(WindowBase):
	"""
	Sequence Processor window for running sequential commands and processes.
	"""
	_persistent_fields: list[str]
	accepted_input_types: list[IOTypes]
	outputs: dict[str, IOTypes]
	connections: dict[str, list[Any]]
	sequence_input_tag: str
	table_tag: str
	run_seq_tag: str
	kill_seq_tag: str
	seq_remaining_tag: str
	seq_progress_bar_tag: str
	table_group_tag: str
	task: threading.Thread | None
	cmd_list: list[str]
	total_duration: int | float
	initial_total_duration: int | float
	seq_running: bool
	seq_error: bool
	check_delay: int
	current_folder: str | None
	raw_sequence: str
	last_sequence: list[str]

	def __init__(self,
				label: str = "Sequence processor",
				win_width: int = 300,
				win_height: int = 200,
				pos: tuple[int, int] = (10, 10),
				uuid: str | None = None,
				outputs: dict[str, Any] | None = None,
				visible: bool = True) -> None: 

		super().__init__(label=label,pos=pos,win_width=win_width,win_height=win_height,uuid=uuid,outputs=outputs,visible=visible)

		self.outputs = outputs or {}
		
		# Ensure CMD dict 2 output is always present to preserve compatibility
		defaults = {
			"CMD dict": IOTypes.CMD_DICT,
			"CMD dict 2": IOTypes.CMD_DICT,
			"Trigger": IOTypes.TRIGGER,
		}
		for k, v in defaults.items():
			if k not in self.outputs:
				self.outputs[k] = v
				
		self.connections = {k: [] for k in self.outputs}

		self._persistent_fields = ["label"]
		self.accepted_input_types = [IOTypes.CMD_LIST, IOTypes.FOLDER_PATH, IOTypes.TEXT, IOTypes.CMD_DICT, IOTypes.TRIGGER]
		
		self.sequence_input_tag = f"seq_writer_input_{self.UUID}"
		self.table_tag = f"seq_writer_table_{self.UUID}"
		self.run_seq_tag = f"seq_run_{self.UUID}"
		self.kill_seq_tag = f"seq_kill_{self.UUID}"
		self.seq_remaining_tag = f"seq_remaining_{self.UUID}"
		self.seq_progress_bar_tag = f"seq_progress_bar_{self.UUID}"
		self.table_group_tag = f"seq_table_group_{self.UUID}"

		self.task = None
		self.cmd_list = []
		self.total_duration = 0
		self.initial_total_duration = 0
		self.seq_running = False
		self.seq_error = False
		self._ready_event = threading.Event()
		self.check_delay = 5 # Check every 5 seconds if the task is still running
		self.current_folder = None
		self.raw_sequence = ""
		
		with dpg.window(label=self.label,
						width=self.win_width,
						height=self.win_height,
						pos=self.pos,
						tag=self.winID,
						show=self.visible):

				with dpg.group(horizontal=True) :
					dpg.add_button(label="Run seq.", tag = self.run_seq_tag,  callback=self.run_sequence)
					dpg.add_button(label="KILL seq.", tag = self.kill_seq_tag,  callback=self.kill_sequence)

				dpg.add_text("Remaining : 00:00:00", tag=self.seq_remaining_tag)
				dpg.add_progress_bar(tag=self.seq_progress_bar_tag, default_value=0.0, width=-1)

				with dpg.group(tag=self.table_group_tag):
					pass

		self.update_permission()
	
	def update_permission(self) -> None:
		"""
		Adapt UI based on app_state.mode.
		Disable closing and lock all controls in user mode.
		"""
		from core.app_state import app_state
		mode = app_state.mode
		is_user = (mode == "user")
		
		if dpg.does_item_exist(self.winID):
			dpg.configure_item(self.winID, no_close=is_user)

		# Elements to toggle
		elements = [
			self.run_seq_tag,
		]
		
		for tag in elements:
			if dpg.does_item_exist(tag):
				if is_user:
					dpg.hide_item(tag)
				else:
					dpg.show_item(tag)

	def is_ready(self) -> bool:
		"""Check if the window is ready to process inputs."""
		return True
	
	def is_outputs_ready(self) -> bool:
		for output in self.outputs:
			is_ready = getattr(output, "is_ready", None)
			if callable(is_ready) and not is_ready():
				return False
		return True
	
	def validate_sequence(self, cmd_list: list[str]) -> None:
		self.seq_error = False
		
		g_commands = set()
		for cmd in cmd_list:
			clean_cmd = cmd
			if cmd.lower().startswith("out2 "):
				clean_cmd = cmd[5:].strip()
			match = re.match(r'(?i)^g(.+)', clean_cmd)
			if match:
				g_commands.add(match.group(1).strip())
		
		if not g_commands:
			logger.error("No group defined (G command missing). Sequence must have at least one group.")
			self.seq_error = True
				
		conflict_groups = []
		if self.current_folder and os.path.exists(self.current_folder) and g_commands:
			existing_files = os.listdir(self.current_folder)
			for g_cmd in g_commands:
				for f in existing_files:
					if f"_{g_cmd}_" in f or f.startswith(f"{g_cmd}_") or f"G{g_cmd}" in f:
						conflict_groups.append(g_cmd)
						break
						
		self._finalize_validation(cmd_list, conflict_groups, missing_group=not g_commands)

	def _finalize_validation(self, cmd_list: list[str], conflict_groups: list[str] | None = None, missing_group: bool = False, show_window: bool = True) -> None:
		if conflict_groups is None:
			conflict_groups = []
			
		if show_window and dpg.does_item_exist(self.winID):
			dpg.configure_item(self.winID, show=True)
			dpg.focus_item(self.winID)

		dpg.push_container_stack(self.table_group_tag)
		if dpg.does_item_exist(self.table_tag):
			dpg.delete_item(self.table_tag)
		
		# Generate a unique tag to avoid DearPyGUI internal state persistence
		import uuid as uuid_module
		self.table_tag = f"seq_writer_table_{self.UUID}_{uuid_module.uuid4().hex[:8]}"
		
		with dpg.table(header_row=True, resizable=True, tag=self.table_tag, borders_innerH=True, borders_innerV=True, borders_outerH=True, borders_outerV=True,policy=dpg.mvTable_SizingStretchProp):
			dpg.add_table_column(label="Cmd", width_stretch=True, init_width_or_weight=0.3)
			dpg.add_table_column(label="Descr", width_stretch=True, init_width_or_weight=0.7)
			self.total_duration = 0
			self.last_sequence = []

			if missing_group:
				with dpg.table_row():
					dpg.add_text("ERROR")
					dpg.add_text("Missing Group (G command)")
				dpg.highlight_table_row(self.table_tag, 0, [255, 0, 0, 50])
				row_offset = 1
			else:
				row_offset = 0

			for i,cmd in enumerate(cmd_list):
				with dpg.table_row():
					dpg.add_text(cmd)
					cmd_result = sequence_processor.trad_cmd(cmd)

					if not cmd_result["valid"]:
						dpg.highlight_table_row(self.table_tag, i + row_offset,  [255,0,0,50])
						self.seq_error = True
						logger.error(f"Invalid command at index {i} ! Please correct : {cmd}")
					else:
						clean_cmd = cmd
						if cmd.lower().startswith("out2 "):
							clean_cmd = cmd[5:].strip()
						match = re.match(r'(?i)^g(.+)', clean_cmd)
						if match:
							group_name = match.group(1).strip()
							if group_name in conflict_groups:
								dpg.highlight_table_row(self.table_tag, i + row_offset,  [255,0,0,50])
								dpg.add_text(f"ERROR: Group already exist")
								self.seq_error = True
								logger.error(f"Group '{group_name}' already exists in folder {self.current_folder}")
								continue

					self.total_duration += cmd_result["duration"]
					dpg.add_text(cmd_result["description"])
			dpg.set_value(self.seq_remaining_tag, f"Remaining : {datetime.timedelta(seconds=int(self.total_duration))}")
			self.initial_total_duration = self.total_duration
			dpg.set_value(self.seq_progress_bar_tag, 0.0)

			if not self.seq_error:
				logger.info(f"Sequence validated with {len(cmd_list)} commands, total duration: {self.total_duration} seconds")
				self.last_sequence = cmd_list

		dpg.pop_container_stack()

	def run_sequence(self, *args: Any, **kwargs: Any) -> None:
		if self.seq_running:
			from core.tutorial_manager import tutorial_manager
			tutorial_manager.play_center_hint("A sequence is already running. Please cancel the current sequence before starting a new one.", type="error_hint", icon_type="warning")
			logger.warning("A sequence is already running, please wait for it to finish or kill it before starting a new one.")
			return
		
		if hasattr(self, 'last_sequence') and self.last_sequence:
			self.validate_sequence(self.last_sequence)

		if not self.seq_error :

			def task(final_group: str | None = None, manual: bool = False) -> None:
				def cleanup(status: str) -> None:
					self.trigger_cb("CMD dict", {"record": False, "save": False, "add_timestamp": False, "manual": False, "export": True})
					dpg.set_value(self.seq_remaining_tag, f"Remaining : {status}")
					if status == "DONE":
						dpg.set_value(self.seq_progress_bar_tag, 1.0)
					self.trigger_cb("Trigger", "stop")
					self.seq_running = False

					if status == "DONE" and not manual:
						reload_cmd = {"scan": True}
						if final_group:
							reload_cmd["group"] = final_group
						self.trigger_cb("CMD dict", reload_cmd)

				for i, cmd in enumerate(self.last_sequence):

					if not self.seq_running :
						logger.info("Sequence aborted")
						cleanup("ABORTED")
						return

					dpg.highlight_table_row(self.table_tag, i,  [0,0,255,50])

					cmd_response = sequence_processor.trad_cmd(cmd)

					if cmd_response.get("out2"):
						self.trigger_cb("CMD dict 2", cmd_response["cmd"])
					else:
						self.trigger_cb("CMD dict", cmd_response["cmd"])

					wait_feedback = bool(cmd_response.get("wait_feedback", False))
					duration = float(cmd_response.get("duration", 0.0))

					if wait_feedback:
						self._ready_event.clear()
						start_wait = time.perf_counter()
						# Timeout: wait until ready event OR auto-advance if duration + 2.0s grace period expires
						max_wait = duration + 2.0 if duration > 0 else 3600.0

						while not self._ready_event.is_set():
							if not self.seq_running:
								logger.info("Sequence aborted")
								cleanup("KILLED")
								return

							time.sleep(0.05)
							now = time.perf_counter()
							elapsed = now - start_wait
							if elapsed >= max_wait:
								logger.info(f"Feedback wait elapsed for command '{cmd}' ({elapsed:.1f}s), auto-advancing.")
								break

							# Live update remaining time display while waiting
							if duration > 0:
								time_spent = min(elapsed, duration)
								rem = max(0.0, self.total_duration - time_spent)
								dpg.set_value(self.seq_remaining_tag, f"Remaining : {datetime.timedelta(seconds=int(rem))}")
								if self.initial_total_duration > 0:
									progress = max(0.0, min(1.0, 1.0 - (rem / self.initial_total_duration)))
									dpg.set_value(self.seq_progress_bar_tag, progress)

						if duration > 0:
							self.total_duration -= duration
							if self.total_duration < 0:
								self.total_duration = 0

						dpg.set_value(self.seq_remaining_tag, f"Remaining : {datetime.timedelta(seconds=int(self.total_duration))}")
						if self.initial_total_duration > 0:
							progress = max(0.0, min(1.0, 1.0 - (self.total_duration / self.initial_total_duration)))
							dpg.set_value(self.seq_progress_bar_tag, progress)

					elif duration > 0:
						remaining_time = duration
						while remaining_time > 0:
							sleep_time = min(1, remaining_time)
							time.sleep(sleep_time)
							remaining_time -= sleep_time

							self.total_duration -= sleep_time
							if self.total_duration < 0:
								self.total_duration = 0

							dpg.set_value(self.seq_remaining_tag, f"Remaining : {datetime.timedelta(seconds=int(self.total_duration))}")
							if self.initial_total_duration > 0:
								progress = 1.0 - (self.total_duration / self.initial_total_duration)
								dpg.set_value(self.seq_progress_bar_tag, progress)

							if not self.seq_running:
								logger.info("Sequence aborted")
								cleanup("KILLED")
								return

					dpg.highlight_table_row(self.table_tag, i,  [0,255,0,50])

				cleanup("DONE")

			if not self.seq_running:
				# Pre-scan sequence for Triggers and Groups using command registry definitions
				current_group = None
				current_A = 0
				names_to_send = []
				has_captures = False
				is_manual = False

				for cmd_str in self.last_sequence:
					clean_cmd = cmd_str
					if cmd_str.lower().startswith("out2 "):
						clean_cmd = cmd_str[5:].strip()
					
					g_match = re.match(r'(?i)^g(.+)', clean_cmd)
					if g_match:
						current_group = g_match.group(1).strip()

					# Track A command
					a_match = re.fullmatch(r'[aA](\d+)', clean_cmd)
					if a_match:
						current_A = int(a_match.group(1))

					# Get translated command
					cmd_result = sequence_processor.trad_cmd(cmd_str)
					captures = cmd_result.get("captures", [])

					if cmd_result.get("cmd", {}).get("manual", False):
						is_manual = True

					if captures:
						has_captures = True
						for cap_name in captures:
							if current_group:
								names_to_send.append(f"{len(names_to_send)}_{current_group}_A{current_A}_{cap_name}")
							else:
								names_to_send.append(f"{len(names_to_send)}_A{current_A}_{cap_name}")

				setup_cmd = {}
				if has_captures:
					setup_cmd["trigger_mode"] = "Trigger"

				if names_to_send:
					setup_cmd["names"] = names_to_send
					setup_cmd["record"] = True

				if not is_manual:
					setup_cmd["record"] = True
					setup_cmd["save"] = True
					setup_cmd["add_timestamp"] = True

				setup_cmd["reset_ts"] = True
				setup_cmd["clear"] = True

				if current_group:
					setup_cmd["group"] = current_group

				setup_cmd["manual"] = is_manual

				self.trigger_cb("CMD dict", setup_cmd)
				time.sleep(0.1)

				# Save sequence JSON
				logger.info(f"SequenceProcessorWin: Attempting to save sequence JSON. current_group='{current_group}', current_folder='{self.current_folder}'")
				if current_group and self.current_folder:
					import json
					json_path = os.path.join(self.current_folder, f"{current_group}.json")
					try:
						# Use the exact raw sequence string for archiving
						with open(json_path, 'w') as f:
							json.dump({
								"sequence": self.raw_sequence,
								"user": app_state.get("username", ""),
								"detector": app_state.get("detector", None)
							}, f, indent=4)
						logger.info(f"Sequence saved to {json_path}")
					except Exception as e:
						logger.error(f"Failed to save sequence JSON: {e}")
				else:
					logger.warning(f"SequenceProcessorWin: Cannot save JSON because current_group or current_folder is missing (current_group={current_group}, current_folder={self.current_folder})")

				self.set_table_color((0, 0, 0, 0))
				self.trigger_cb("Trigger", "start")
				self.seq_running = True
				self.task = threading.Thread(target=task, args=(current_group, is_manual))
				self.task.start()
			else :
				from core.tutorial_manager import tutorial_manager
				tutorial_manager.play_center_hint("A sequence is already running. Please cancel the current sequence before starting a new one.", type="error_hint", icon_type="warning")
				logger.warning("A sequence is already running, please wait for it to finish or kill it before starting a new one.")
		else:
			from core.tutorial_manager import tutorial_manager
			tutorial_manager.play_center_hint("Sequence validation failed. Please check for errors (missing group or group already exists) before running.", type="error_hint", icon_type="error")
			logger.error("Cannot run a sequence that contains errors. Please validate the sequence first.")
			dpg.set_value(self.seq_remaining_tag, f"Remaining : ERROR")

	def set_table_color(self, rgba: tuple[int, int, int, int] = (0, 0, 0, 0)) -> None:
		if hasattr(self, "table_tag") and self.table_tag and dpg.does_item_exist(self.table_tag):
			try:
				rows = dpg.get_item_children(self.table_tag, 1)
				if rows:
					for i in range(len(rows)):
						dpg.highlight_table_row(self.table_tag, i, rgba)
			except Exception as e:
				logger.error(f"Failed to set table color: {e}")

	def kill_sequence(self, *args: Any, **kwargs: Any) -> None:
		self.seq_running = False
		self.set_table_color((255, 0, 0, 50))

	def clear_sequence(self, *args: Any, **kwargs: Any) -> None:
		"""Clear the sequence processor state."""
		logger.info(f"SequenceProcessorWin.clear_sequence called (current_folder before clear: {self.current_folder})")
		self.seq_running = False
		self.seq_error = False
		self.raw_sequence = ""
		self.last_sequence = []
		self.cmd_list = []
		self.task = None
		# Preserve current_folder to avoid losing the target directory during sequence validation/clearing
		logger.info(f"SequenceProcessorWin: preserving current_folder: {self.current_folder}")
		self.total_duration = 0
		self.initial_total_duration = 0
		self._finalize_validation([], show_window=False)
		logger.info("Sequence processor cleared")

	def input_cb(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
		if kwargs.get("data_type") == IOTypes.FOLDER_PATH or (isinstance(data, str) and os.path.isdir(data)):
			self.current_folder = data
			logger.info(f"Sequence Processor set current_folder: {self.current_folder}")
			return
			
		data_type = kwargs.get("data_type")
		data_content = kwargs.get("data") if "data" in kwargs else data

		# 1. Handle READY / Trigger feedback signals from downstream modules
		if data_type == IOTypes.TRIGGER or (isinstance(data_content, str) and data_content.upper() in ["READY", "RECORD_STOP", "DONE", "STOP", "FINISHED"]):
			logger.info(f"Sequence Processor {self.UUID} received feedback trigger: '{data_content}'")
			self._ready_event.set()
			return

		if data_type == IOTypes.CMD_DICT or isinstance(data_content, dict):
			if isinstance(data_content, dict) and (data_content.get("ready") or data_content.get("done") or data_content.get("finished")):
				logger.info(f"Sequence Processor {self.UUID} received feedback dict: {data_content}")
				self._ready_event.set()
				return
			if data_content.get("start_sequence"):
				self.run_sequence()
			if data_content.get("clear", False):
				self.clear_sequence()
			return

		# Block setting sequence if a sequence is already running
		if self.seq_running:
			if data_type == IOTypes.TEXT or isinstance(data_content, str) or isinstance(data_content, list):
				from core.tutorial_manager import tutorial_manager
				tutorial_manager.play_center_hint("A sequence is already running. Please cancel the current sequence before setting a new one.", type="error_hint", icon_type="warning")
				logger.warning("A sequence is already running, ignoring incoming sequence update.")
				return

		if data_type == IOTypes.TEXT or isinstance(data_content, str):
			self.raw_sequence = data_content
			cmd_list = sequence_processor.get_commands(self.raw_sequence)
			self.validate_sequence(cmd_list)
			return

		if isinstance(data_content, list):
			# Backward compatibility or direct list input (won't have raw original formatting)
			self.raw_sequence = "\n".join(data_content)
			self.validate_sequence(data_content)

	def trigger_cb(self, target: str | int = "CMD dict", cmd: dict[str, Any] | str = {}) -> None:
		"""Sends the last known status as STATUS_DICT to all outputs."""
		if isinstance(target, int):
			if target == 0:
				output_key = "CMD dict"
			elif target == 1:
				output_key = "Trigger"
			else:
				output_key = "CMD dict 2"
		else:
			output_key = target

		connected_modules = self.connections.get(output_key, [])
		for module in connected_modules:
			if output_key == "Trigger":
				module.input_cb(cmd, data_type=IOTypes.TRIGGER)
			else:
				module.input_cb(cmd)

EXPORTED_CLASS = SequenceProcessorWin
EXPORTED_NAME = "Sequence processor"
