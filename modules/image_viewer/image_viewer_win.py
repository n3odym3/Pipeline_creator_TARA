import dearpygui.dearpygui as dpg
import numpy as np
import threading
import time
import cv2
import os
from loguru import logger

from core.window_base import WindowBase
from modules.image_viewer.clipboard_injector import clipboardinjector
from core.input_output_types import IOTypes

class Image_viewer_win(WindowBase):
	"""
	Image_viewer_win:
	A DearPyGui-based modular window to display images with support for:
	- grayscale rescaling
	- color palette application
	- FPS measurement
	- annotated scale overlay
	- clipboard/image export
	- input handling from various sources (file, folder, ndarray)
	- ROI selection and management
	"""

	def __init__(self,
				label="Image Viewer",
				win_width=1280,
				win_height=900,
				pos=(100, 100),
				uuid=None,
				outputs=None,
				visible=True,
				colorize=True,
				intensity_rescaling=True,
				fps_counter=True,
				image_export=True,
				image_depth="12bit",
                show_scale=True,
                categories_count=5,
                use_rgba=False,
                export_path=None,
                **kwargs):

		self.use_rgba = use_rgba

		super().__init__(label=label, pos=pos, win_width=win_width, win_height=win_height,
			uuid=uuid, outputs=outputs or [], visible=visible, **kwargs)

		# FPS tracking - separate input and display
		# FPS tracking
		self.fps = 0
		self.fps_now = time.time()
		self.fps_last = self.fps_now

		# Lock for threaded updates
		self.lock = threading.Lock()

		# Last image reference
		self.last_image = None

		# Display options
		self.negative = False
		self.palette = "Inferno"
		self.imgcount = 0
		
		# Performance optimization: frame counter for FPS display
		self._frame_count = 0
		self._fps_update_interval = 10  # Update FPS display every N frames

		# Texture state
		self.texture_width = 0
		self.texture_height = 0

		# DPG element tags
		self.texture_tag = f"viewer_texture_{self.UUID}"
		self.plot_tag = f"viewer_plot_{self.UUID}"
		self.fps_tag = f"viewer_fps_{self.UUID}"
		self.name_tag = f"viewer_name_{self.UUID}"
		self.count_tag = f"viewer_count_{self.UUID}"
		self.save_btn = f"viewer_save_{self.UUID}"
		self.auto_remap_tag = f"viewer_autoremap_{self.UUID}"
		self.show_scale_tag = f"viewer_showscale_{self.UUID}"
		self.inf_tag = f"viewer_inf_{self.UUID}"
		self.sup_tag = f"viewer_sup_{self.UUID}"
		self.texture_reg_tag = f"viewer_tex_reg_{self.UUID}"
		self.rgba_checkbox_tag = f"viewer_rgba_{self.UUID}"
		dpg.add_texture_registry(tag=self.texture_reg_tag, show=False)

		# ROI tags
		self.roi_enable_tag = f"viewer_roi_enable_{self.UUID}"
		self.roi_w_tag = f"viewer_roi_w_{self.UUID}"
		self.roi_h_tag = f"viewer_roi_h_{self.UUID}"
		self.roi_thresh_tag = f"viewer_roi_thresh_{self.UUID}"
		self.roi_show_mask_tag = f"viewer_roi_showmask_{self.UUID}"
		self.roi_live_crop_tag = f"viewer_roi_livecrop_{self.UUID}"
		self.roi_result_tag = f"viewer_roi_result_{self.UUID}"
		
		# Tags for permission management
		self.mode_combo_tag = f"viewer_mode_combo_{self.UUID}"
		self.btn_clear_rois_tag = f"viewer_clear_rois_{self.UUID}"
		self.export_group_tag = f"viewer_export_group_{self.UUID}"
		self.fps_group_tag = f"viewer_fps_group_{self.UUID}"
		self.roi_thresh_group_tag = f"viewer_roi_thresh_group_{self.UUID}"
		self.palette_combo_tag = f"viewer_palette_combo_{self.UUID}"
		self.depth_combo_tag = f"viewer_depth_combo_{self.UUID}"
		
		# Category mapping tags
		self.cat_group_tag = f"viewer_cat_group_{self.UUID}"
		self.cat_count_tag = f"viewer_cat_count_{self.UUID}"
		self.btn_send_palette_tag = f"viewer_send_palette_{self.UUID}"
		self.btn_send_result_tag = f"viewer_send_result_{self.UUID}"
		
		# Overlay drawing tags
		self.roi_draw_layer = f"viewer_roi_draw_{self.UUID}"

		# Interactive state
		self.mouse_x = 0
		self.mouse_y = 0
		self.is_hovering = False

		# Config flags
		self.colorize = colorize
		self.intensity_rescaling = intensity_rescaling
		self.fps_counter = fps_counter
		self.image_export = image_export
		self.image_depth = image_depth
		self.show_scale = show_scale
		self.export_path = export_path
		self.categories_count = 5

		# ROI Storage
		self.rois = {} # {uuid: {rect: [x,y,w,h], name: str, selected: bool, ...}}
		self.last_selected_roi_uuid = None
		self.roi_index_counter = 1

		# Persistence and IO setup
		self._persistent_fields = [
			"label", "colorize", "intensity_rescaling",
			"fps_counter", "image_export", "image_depth", "show_scale", "export_path", "categories_count", "use_rgba"
		]
		self.accepted_input_types = [
			IOTypes.FRAME, IOTypes.FILE_PATH, IOTypes.FRAME12, IOTypes.FRAME16,
			IOTypes.ROI_SAMPLE, IOTypes.CMD_DICT, IOTypes.FOLDER_PATH,
			IOTypes.FRAME_MASK_PAIR, IOTypes.MASK
		]
		self.output_types = [IOTypes.FRAME, IOTypes.ROI_SAMPLE, IOTypes.FRAME, IOTypes.FRAME]

		self.outputs = {
			"Frame": IOTypes.FRAME,
			"ROI Data": IOTypes.ROI_SAMPLE,
			"Cropped Frame": IOTypes.FRAME,
			"Lowdepth Frame": IOTypes.FRAME
		}
		self.connections = {k: [] for k in self.outputs}
		# Rendering State
		self.running = True

		# Threading & Latest Frame approach (no blocking, 600+ FPS)
		self.latest_frame = None
		self.frame_lock = threading.Lock()
		self.update_event = threading.Event()

		self._build_interface(win_width, win_height, pos)
		self.update_permission()

		# Start Render Thread
		self.worker_thread = threading.Thread(target=self._render_loop, daemon=True)
		self.worker_thread.start()

	def update_permission(self):
		"""
		Adapt UI based on app_state.mode.
		In 'user' mode: hide normal/negative combo, show selection, live crop, clear, threshold, export group.
		"""
		from core.app_state import app_state
		mode = app_state.mode
		
		# Elements to toggle
		elements = [
			self.mode_combo_tag,
			self.roi_show_mask_tag,
			self.roi_live_crop_tag,
			self.btn_clear_rois_tag,
			self.roi_thresh_group_tag,
			self.export_group_tag,
			self.show_scale_tag,
			self.fps_group_tag,
			self.palette_combo_tag,
			self.depth_combo_tag,
			self.roi_enable_tag,
			self.cat_group_tag
		]
		
		show = (mode != "user")
		
		for tag in elements:
			if dpg.does_item_exist(tag):
				if show:
					dpg.show_item(tag)
				else:
					dpg.hide_item(tag)

		is_user = (mode == "user")
		dpg.configure_item(self.winID, no_close=is_user)

	def _build_interface(self, win_width, win_height, pos):
		"""Creates the DearPyGui layout."""
		with dpg.window(label=self.label, width=win_width, height=win_height,
			pos=pos, tag=self.winID, show=self.visible):

			with dpg.group(horizontal=True):
				if self.colorize:
					dpg.add_combo(("Normal", "Negative"), tag=self.mode_combo_tag, width=120, default_value="Normal", callback=self.set_mode)
					dpg.add_combo(("B&W", "Inferno", "Jet", "HSV"), tag=self.palette_combo_tag, width=120, default_value=self.palette, callback=self.set_mode)
					dpg.add_combo(("8bit", "12bit", "16bit_norm", "16bit_abs"), tag=self.depth_combo_tag, width=120, default_value=self.image_depth, callback=self.set_depth)

				if self.intensity_rescaling:
					dpg.add_text("Auto:")
					dpg.add_checkbox(default_value=True, tag=self.auto_remap_tag, callback=self.update_image_callback)
					dpg.add_text("Scale:")
					dpg.add_checkbox(default_value=self.show_scale, tag=self.show_scale_tag, callback=self.update_image_callback)
					dpg.add_text("Inf:")
					dpg.add_drag_int(default_value=0, min_value=0, max_value=255, width=70, tag=self.inf_tag, callback=self.update_image_callback)
					dpg.add_text("Sup:")
					dpg.add_drag_int(default_value=255, min_value=0, max_value=255, width=70, tag=self.sup_tag, callback=self.update_image_callback)

				if self.fps_counter:
					with dpg.group(horizontal=True, tag=self.fps_group_tag):
						dpg.add_text("| FPS:")
						dpg.add_text("0", tag=self.fps_tag)

				dpg.add_checkbox(label="RGBA", default_value=self.use_rgba, tag=self.rgba_checkbox_tag, callback=self._on_toggle_rgba)

			dpg.add_separator()
			with dpg.group(horizontal=True):
				dpg.add_checkbox(label="Enable ROI", default_value=True, tag=self.roi_enable_tag, callback=self.update_image_callback)
				dpg.add_checkbox(label="Show Selection", tag=self.roi_show_mask_tag, callback=self.update_image_callback)
				dpg.add_checkbox(label="Live Crop", tag=self.roi_live_crop_tag, default_value=False)
				dpg.add_button(label="Clear", tag=self.btn_clear_rois_tag, callback=self.clear_rois)
			
				dpg.add_text("W:")
				dpg.add_drag_int(width=60, tag=self.roi_w_tag, default_value=25, min_value=1, callback=self.update_image_callback)
				dpg.add_text("H:")
				dpg.add_drag_int(width=60, tag=self.roi_h_tag, default_value=25, min_value=1, callback=self.update_image_callback)

				with dpg.group(horizontal=True, tag=self.roi_thresh_group_tag):
					dpg.add_text("Threshold %:")
					dpg.add_drag_int(width=100, tag=self.roi_thresh_tag, default_value=100, min_value=1, max_value=100, callback=self.update_image_callback)
				dpg.add_text("Mean Value:", color=(255, 100, 100))
				dpg.add_text("0.0", tag=self.roi_result_tag)

			if self.image_export:
				with dpg.group(horizontal=True, tag=self.export_group_tag):
					dpg.add_button(label="Clipboard", callback=self.copy_to_clipboard)
					dpg.add_input_text(default_value="Image", width=150, tag=self.name_tag, callback=self.reset_counter)
					dpg.add_text("count: 0", tag=self.count_tag)
					dpg.add_button(label="Save", tag=self.save_btn, callback=self.save_image)
					
					self.save_tooltip_text_tag = f"viewer_save_tooltip_text_{self.UUID}"
					with dpg.tooltip(parent=self.save_btn):
						initial_tip = f"Saves to: {self.export_path}" if self.export_path else "Saves to default folder"
						dpg.add_text(initial_tip, tag=self.save_tooltip_text_tag)

			# Category mapping group
			with dpg.group(horizontal=True, tag=self.cat_group_tag):
				dpg.add_text("Categories:")
				dpg.add_input_int(default_value=self.categories_count, min_value=2, max_value=20, step=1, width=80, tag=self.cat_count_tag, callback=self._update_categories_count)
				dpg.add_button(label="Send Palette", tag=self.btn_send_palette_tag, callback=self.send_palette_to_clipboard)
				dpg.add_button(label="Send Result", tag=self.btn_send_result_tag, callback=self.send_result_to_clipboard)
			
			# Add mouse handler for ROI interaction
			with dpg.handler_registry():
				dpg.add_mouse_move_handler(callback=self.on_mouse_move)
				dpg.add_mouse_click_handler(callback=self.on_mouse_click)

			self.init_viewer(1000, 1000)
			self._apply_transparency_theme()

	def on_mouse_move(self, sender, data, *args, **kwargs):
		"""Tracks mouse position for ROI."""
		try:
			if dpg.does_item_exist(self.plot_tag) and dpg.is_item_hovered(self.plot_tag):
				self.is_hovering = True
				# This returns (x, y) in plot coords
				mpos = dpg.get_plot_mouse_pos()
				self.mouse_x = int(mpos[0])
				self.mouse_y = int(mpos[1])
				
				if dpg.get_value(self.roi_show_mask_tag) and self.last_image is not None:
					self.update_image(self.last_image)
				
				self._update_roi_overlay()
			else:
				self.is_hovering = False
				# Hide ROI when leaving plot
				self._update_roi_overlay()
		except Exception:
			# DPG might throw SystemError if item state is invalid during resize/reinit
			self.is_hovering = False

	def on_mouse_click(self, sender, data, *args, **kwargs):
		"""Handles mouse clicks, specifically CTRL+Click or Middle Click for ROI creation."""
		is_ctrl_click = (data == 0 and dpg.is_key_down(dpg.mvKey_LControl))
		is_middle_click = (data == 2)
		
		if (is_ctrl_click or is_middle_click) and self.is_hovering and dpg.get_value(self.roi_enable_tag):
			# Create new ROI
			roi_uuid = dpg.generate_uuid()
			name = str(self.roi_index_counter)
			self.roi_index_counter += 1
			
			w = dpg.get_value(self.roi_w_tag)
			h = dpg.get_value(self.roi_h_tag)
			
			# Center logic
			x = self.mouse_x - w // 2
			y = self.mouse_y - h // 2
			
			# Store ROI
			self.rois[roi_uuid] = {
				"uuid": roi_uuid,
				"name": name,
				"rect": [x, y, w, h],
				"selected": True,
				"threshold": dpg.get_value(self.roi_thresh_tag)
			}
			
			# Emit
			self.last_selected_roi_uuid = roi_uuid
			self._emit_roi(self.rois[roi_uuid], action="create")
			self._update_roi_overlay()
			if self.last_image is not None:
				self._emit_cropped_frame(self.last_image, force_emit=True)

	def _emit_roi(self, roi_data, action="select"):
		"""Emits ROI_SAMPLE to outputs."""
		if action == "clear":
			sample = {"action": "clear", "clear": True}
		else:
			sample = {
				"uuid": roi_data["uuid"],
				"name": roi_data["name"],
				"rect": roi_data["rect"],
				"threshold": roi_data.get("threshold", 100),
				"action": action
			}
		
		# Send to all connected outputs
		for module in self.connections.get("ROI Data", []):
			module.input_cb(sample=sample, data_type=IOTypes.ROI_SAMPLE)

	def clear_rois(self, *args, **kwargs):
		"""Clears all ROIs from the viewer and notifies connected modules."""
		self._handle_roi_input({"clear": True})
		self._emit_roi(None, action="clear")
		self.roi_index_counter = 1
		self.last_selected_roi_uuid = None
		self.update_image_callback()

	def input_cb(self, *args, **kwargs):
		"""Receives and processes image input from args or kwargs (supports ndarray or filepath)."""
		# Handle ROI Input
		if "data_type" in kwargs and kwargs["data_type"] == IOTypes.ROI_SAMPLE:
			self._handle_roi_input(kwargs["sample"])
			return

		if "data_type" in kwargs and kwargs["data_type"] == IOTypes.CMD_DICT:
			cmd = kwargs.get("data") if kwargs.get("data") is not None else kwargs.get("sample") # Fallback to sample if used by sender
			if isinstance(cmd, dict):
				if cmd.get("reset_counter"):
					self.roi_index_counter = 1
				if cmd.get("clear") or cmd.get("action") == "clear":
					self._handle_roi_input({"clear": True})
			return

		# Handle FOLDER_PATH input (either via explicit data_type or if the data/args[0] is a directory string)
		data_type = kwargs.get("data_type")
		path_val = kwargs.get("path")
		if path_val is None:
			path_val = kwargs.get("data")
		if path_val is None:
			path_val = args[0] if (args and not isinstance(args[0], np.ndarray)) else None

		if data_type == IOTypes.FOLDER_PATH or (isinstance(path_val, str) and os.path.isdir(path_val)):
			if isinstance(path_val, str):
				self.export_path = path_val
				logger.info(f"Image Viewer export path set to: {self.export_path}")
				self.update_save_tooltip()
				return

		frame = kwargs["data"] if "data" in kwargs else (args[0] if args else None) #Numpy compatible evaluation

		# Handle tuple input (frame, name)
		if isinstance(frame, (tuple, list)) and len(frame) >= 1:
			if isinstance(frame[0], np.ndarray):
				frame = frame[0]
				# We could also use frame[1] as name if needed, but Viewer doesn't use it yet (except maybe for saving?)

		if isinstance(frame, str) and os.path.exists(frame) and not os.path.isdir(frame):
			frame = cv2.imread(frame, cv2.IMREAD_UNCHANGED)

		if isinstance(frame, np.ndarray):
			self.update_image(frame)

	def convert_to_lowdepth(self, frame: np.ndarray) -> np.ndarray:
		"""Converts high bit-depth image to 8-bit based on image_depth."""
		match self.image_depth:
			case "8bit":
				return frame
			case "12bit":
				return (frame >> 4).astype(np.uint8)
			case "16bit_norm":
				return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
			case "16bit_abs":
				return (frame >> 8).astype(np.uint8)

	def get_minmax_values(self, frame: np.ndarray) -> tuple[int, int]:
		"""Returns min and max intensity values. Optimized with OpenCV."""
		if frame is None or frame.size == 0:
			return 0, 255

		if frame.ndim == 2:
			min_val, max_val, _, _ = cv2.minMaxLoc(frame)
		else:
			# For multi-channel images, compute global min/max
			min_val = np.min(frame)
			max_val = np.max(frame)
		
		return int(min_val), int(max_val)

	def _lowdepth_to_original(self, val_8bit, frame):
		"""Converts an 8-bit value (from sliders) back to the original image depth range."""
		if frame is None:
			return val_8bit
			
		match self.image_depth:
			case "8bit":
				return val_8bit
			case "12bit":
				return val_8bit << 4
			case "16bit_abs":
				return val_8bit << 8
			case "16bit_norm":
				fmin, fmax = self.get_minmax_values(frame)
				return fmin + (val_8bit / 255.0) * (fmax - fmin)
		return val_8bit

	def _get_current_scale_limits(self):
		"""Returns (minval, maxval) in original precision based on UI state (Auto vs Manual)."""
		if self.last_image is None:
			return 0, 255
			
		if dpg.get_value(self.auto_remap_tag):
			sample = self.last_image[::5, ::5]
			minval = int(np.percentile(sample, 1))
			maxval = int(np.max(sample))
			return minval, maxval
		else:
			ui_inf = dpg.get_value(self.inf_tag)
			ui_sup = dpg.get_value(self.sup_tag)
			minval = self._lowdepth_to_original(ui_inf, self.last_image)
			maxval = self._lowdepth_to_original(ui_sup, self.last_image)
			return minval, maxval

	def remap_pixel_intensity(self, frame: np.ndarray, low: float, high: float) -> np.ndarray:
		"""Normalizes frame between [low, high] and scales to 8-bit using fast OpenCV scaling."""
		if int(low) == 0 and int(high) == 255 and frame.dtype == np.uint8:
			return frame

		diff = high - low
		if diff <= 0: diff = 1.0
		
		# cv2.convertScaleAbs scales and clips to uint8 in a single SIMD-accelerated C++ pass
		alpha = 255.0 / diff
		beta = -low * alpha
		return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

	def add_intensity_scale(self, img: np.ndarray, scale_width=30, position='left') -> np.ndarray:
		"""Adds vertical or horizontal intensity scale bar to image."""
		h = img.shape[0]
		gradient = np.linspace(255, 0, h).reshape((h, 1)).astype(np.uint8)
		scale = np.tile(gradient, (1, scale_width))
		
		if img.ndim == 3:
			# Broadcast scale to matching number of channels
			scale_multichannel = np.stack([scale] * img.shape[2], axis=-1)
			# For RGBA, keep alpha at 255 for the scale bar
			if img.shape[2] == 4:
				scale_multichannel[:, :, 3] = 255
			scale = scale_multichannel

		overlay = img.copy()

		match position:
			case 'right': overlay[:, -scale_width:] = scale
			case 'left': overlay[:, :scale_width] = scale
			case 'top': overlay[:scale_width, :] = scale.T
			case 'bottom': overlay[-scale_width:, :] = scale.T

		return overlay

	def calc_fps(self) -> float:
		"""Computes FPS (how fast frames arrive/render)."""
		self.fps_now = time.time()
		delta = self.fps_now - self.fps_last
		if delta > 0:
			self.fps = 1 / delta
			self.fps_last = self.fps_now
		return self.fps

	def _apply_transparency_theme(self):
		"""Applies or removes transparent plot background theme based on use_rgba."""
		if not hasattr(self, "_plot_transparent_theme") or not dpg.does_item_exist(self._plot_transparent_theme):
			with dpg.theme() as self._plot_transparent_theme:
				with dpg.theme_component(dpg.mvPlot):
					dpg.add_theme_color(dpg.mvPlotCol_PlotBg, (0, 0, 0, 0), category=dpg.mvThemeCat_Plots)
					dpg.add_theme_color(dpg.mvPlotCol_FrameBg, (0, 0, 0, 0), category=dpg.mvThemeCat_Plots)
		
		if dpg.does_item_exist(self.plot_tag):
			if self.use_rgba:
				dpg.bind_item_theme(self.plot_tag, self._plot_transparent_theme)
			else:
				dpg.bind_item_theme(self.plot_tag, 0)

	def _on_toggle_rgba(self, sender, app_data, user_data=None):
		"""Toggles RGBA mode and updates plot transparency."""
		self.use_rgba = bool(app_data)
		if self.texture_width > 0 and self.texture_height > 0:
			self.init_viewer(self.texture_width, self.texture_height)
			if self.last_image is not None:
				with self.frame_lock:
					self.latest_frame = self.last_image
				self.update_event.set()
		else:
			self._apply_transparency_theme()

	def convert_to_texture(self, frame: np.ndarray) -> np.ndarray:
		"""Converts image to flattened RGB or RGBA float array for DearPyGui (optimized with buffer reuse)."""
		if frame.dtype != np.uint8:
			frame = frame.astype(np.uint8)
		
		channels = 4 if self.use_rgba else 3
		expected_size = frame.shape[0] * frame.shape[1] * channels

		if not hasattr(self, '_texture_buffer') or self._texture_buffer.size != expected_size:
			self._texture_buffer = np.zeros(expected_size, dtype=np.float32)

		if self.use_rgba:
			if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
				conv_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGBA)
			elif frame.shape[2] == 4:
				conv_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
			elif frame.shape[2] == 3:
				conv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
			else:
				conv_frame = cv2.cvtColor(frame[:, :, :1], cv2.COLOR_GRAY2RGBA)
		else:
			if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
				conv_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
			elif frame.shape[2] == 3:
				conv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			elif frame.shape[2] == 4:
				conv_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
			else:
				conv_frame = cv2.cvtColor(frame[:, :, :1], cv2.COLOR_GRAY2RGB)

		flattened = conv_frame.ravel()
		
		if len(self._texture_buffer) == len(flattened):
			np.multiply(flattened, 1.0 / 255.0, out=self._texture_buffer, casting='unsafe')
			return self._texture_buffer
		else:
			self._texture_buffer = flattened.astype(np.float32) * (1.0 / 255.0)
			return self._texture_buffer

	def init_viewer(self, width: int, height: int):
		"""Initializes DearPyGui image display area (main thread safe)."""
		from core.automation_manager import automation_manager
		if threading.current_thread() is threading.main_thread():
			self._init_viewer_internal(width, height)
		else:
			automation_manager.post_to_main_thread(self._init_viewer_internal, width, height, wait=True)

	def _init_viewer_internal(self, width: int, height: int):
		if dpg.does_item_exist(self.texture_tag):
			dpg.delete_item(self.texture_tag)
		if dpg.does_item_exist(self.plot_tag):
			dpg.delete_item(self.plot_tag)

		channels = 4 if self.use_rgba else 3
		fmt = dpg.mvFormat_Float_rgba if self.use_rgba else dpg.mvFormat_Float_rgb

		self._texture_buffer = np.zeros(height * width * channels, dtype=np.float32)
		texture_data = self._texture_buffer

		dpg.add_raw_texture(
			width=width,
			height=height,
			default_value=texture_data,
			tag=self.texture_tag,
			format=fmt,
			parent=self.texture_reg_tag,
		)

		plot_tag = dpg.add_plot(
			tag=self.plot_tag,
			no_menus=True,
			no_title=True,
			width=-1,
			height=-1,
			parent=self.winID,
		)
		dpg.add_plot_legend(parent=plot_tag)
		xaxis = dpg.add_plot_axis(dpg.mvXAxis, no_gridlines=True, no_tick_labels=True, no_tick_marks=True, parent=plot_tag)
		yaxis = dpg.add_plot_axis(dpg.mvYAxis, no_gridlines=True, no_tick_labels=True, no_tick_marks=True, parent=plot_tag)
		dpg.add_image_series(self.texture_tag, [0, 0], [width, height], parent=yaxis)
		dpg.fit_axis_data(xaxis)
		dpg.fit_axis_data(yaxis)

		# ROI Overlay initialization
		rect_tag = f"roi_rect_{self.UUID}"
		text_tag = f"roi_text_{self.UUID}"
		if dpg.does_item_exist(rect_tag):
			dpg.delete_item(rect_tag)
		if dpg.does_item_exist(text_tag):
			dpg.delete_item(text_tag)
		dpg.draw_rectangle(pmin=(0, 0), pmax=(0, 0), color=(0, 255, 0, 255), thickness=2, tag=rect_tag, show=False, parent=plot_tag)
		dpg.add_plot_annotation(label="ROI", default_value=(0, 0), color=(0, 255, 0, 255), offset=(10, -10), tag=text_tag, show=False, parent=plot_tag)

		self.texture_width = width
		self.texture_height = height
		self._apply_transparency_theme()

	def colorize_frame(self, frame: np.ndarray) -> np.ndarray:
		"""Applies negative or colormap to the 8-bit image."""
		
		if self.negative:
			frame = 255 - frame

		if frame.ndim == 3 and frame.shape[2] >= 3:
			# Skip colormapping for multi-channel images
			return frame

		match self.palette:
			case 'B&W': return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
			case 'Inferno': return cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO)
			case 'Jet': return cv2.applyColorMap(frame, cv2.COLORMAP_JET)
			case 'HSV': return cv2.applyColorMap(frame, cv2.COLORMAP_HSV)
		return frame

	def remap_and_annotate(self, frame, min_val=None, max_val=None):
		"""Remaps intensities, adds scale bar and min/max text."""
		# Calculate 8-bit limits for the VIEW (sliders)
		if dpg.get_value(self.auto_remap_tag):
			# Optimization: Use strided sampling for percentile to avoid framerate drop
			sample = frame[::5, ::5]
			min_val_8bit = int(np.percentile(sample, 1))
			max_val_8bit = int(np.max(sample))
			dpg.set_value(self.sup_tag, max_val_8bit)
			
			if min_val is None: min_val = min_val_8bit
			if max_val is None: max_val = max_val_8bit
		

		frame = self.remap_pixel_intensity(frame, dpg.get_value(self.inf_tag), dpg.get_value(self.sup_tag))
		
		# Check if scale annotations are enabled
		if dpg.get_value(self.show_scale_tag):
			frame = self.add_intensity_scale(frame)

			if min_val is None or max_val is None:
				min_val, max_val = self.get_minmax_values(frame) 

			frame = self.colorize_frame(frame)
			cv2.putText(frame, f"{max_val:.0f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
			cv2.putText(frame, f"{min_val:.0f}", (30, frame.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
		else:
			# Still colorize if needed, but no text/scale
			frame = self.colorize_frame(frame)

		return frame

	def _handle_roi_input(self, sample):
		"""Handles updates from ROI Manager."""
		if sample.get("clear"):
			# Clear all ROIs
			for r_uuid in list(self.rois.keys()):
				r_tag = f"roi_rect_{self.UUID}_{r_uuid}"
				t_tag = f"roi_text_{self.UUID}_{r_uuid}"
				if dpg.does_item_exist(r_tag): dpg.delete_item(r_tag)
				if dpg.does_item_exist(t_tag): dpg.delete_item(t_tag)
			self.rois.clear()
			self._update_roi_overlay()
			return

		uuid = sample.get("uuid")
		if not uuid:
			return
		
		action = sample.get("action")
		
		if action == "rename":
			if uuid in self.rois:
				self.rois[uuid]["name"] = sample["name"]
				self._update_roi_overlay()
		elif action == "select":
			if uuid in self.rois:
				self.rois[uuid]["selected"] = True
				self.last_selected_roi_uuid = uuid
				self._update_roi_overlay()
				if self.last_image is not None:
					self._emit_cropped_frame(self.last_image, force_emit=True)
		elif action == "unselect":
			if uuid in self.rois:
				self.rois[uuid]["selected"] = False
				self._update_roi_overlay()
		elif action == "delete":
			if uuid in self.rois:
				del self.rois[uuid]
				# Cleanup items
				r_tag = f"roi_rect_{self.UUID}_{uuid}"
				t_tag = f"roi_text_{self.UUID}_{uuid}"
				if dpg.does_item_exist(r_tag): dpg.delete_item(r_tag)
				if dpg.does_item_exist(t_tag): dpg.delete_item(t_tag)
				self._update_roi_overlay()
		elif action == "highlight":
			if uuid in self.rois:
				self.rois[uuid]["highlighted"] = True
				self._update_roi_overlay()
		elif action == "unhighlight":
			if uuid in self.rois:
				self.rois[uuid]["highlighted"] = False
				self._update_roi_overlay()
		elif action == "create":
			if uuid not in self.rois:
				self.rois[uuid] = {
					"uuid": uuid,
					"name": sample.get("name", "ROI"),
					"rect": sample.get("rect", [0, 0, 100, 100]),
					"selected": True,
					"threshold": sample.get("threshold", 100)
				}
				self.last_selected_roi_uuid = uuid
				self._update_roi_overlay()

	def _update_roi_overlay(self):
		"""Updates ROI rectangle and annotation immediately (Main Thread)."""
		
		# 1. Update Persistent ROIs
		for r_uuid, r_data in self.rois.items():
			r_tag = f"roi_rect_{self.UUID}_{r_uuid}"
			t_tag = f"roi_text_{self.UUID}_{r_uuid}"
			
			x, y, w, h = r_data["rect"]
			x1, y1 = x, y
			x2, y2 = x + w, y + h
			
			# Create items if they don't exist
			if not dpg.does_item_exist(r_tag):
				dpg.draw_rectangle(pmin=(x1, y1), pmax=(x2, y2), color=(0, 255, 0, 255), thickness=2, parent=self.plot_tag, tag=r_tag)
				dpg.add_plot_annotation(label=r_data["name"], default_value=(x2, y2), color=(0, 255, 0, 255), offset=(10, -10), parent=self.plot_tag, tag=t_tag)
			
			# Update Appearance
			color = (0, 255, 0, 255) if r_data.get("selected", False) else (0, 100, 0, 255)
			
			# Highlight effect
			fill_color = (0, 255, 0, 70) if r_data.get("highlighted", False) else (0, 0, 0, 0)
			
			# Only show if ROI enabled? Or always show persistent ones? Usually always if they exist.
			# But let's respect the master switch for now or maybe just the preview switch?
			# User requirement: "roi_enable" usually toggles the analysis feature.
			# I will hide them if roi_enable is False to be safe/clean.
			show = dpg.get_value(self.roi_enable_tag)
			
			dpg.configure_item(r_tag, pmin=(x1, y1), pmax=(x2, y2), color=color, fill=fill_color, show=show)
			dpg.configure_item(t_tag, default_value=(x2, y2), label=r_data["name"], color=color, show=show)

		# 2. Update Preview/Hover ROI
		rect_tag = f"roi_rect_{self.UUID}"
		text_tag = f"roi_text_{self.UUID}"
		
		if not dpg.does_item_exist(rect_tag) or not dpg.does_item_exist(text_tag):
			return

		if not dpg.get_value(self.roi_enable_tag) or not self.is_hovering:
			dpg.configure_item(rect_tag, show=False)
			dpg.configure_item(text_tag, show=False)
			return

		# Get current ROI params
		w = dpg.get_value(self.roi_w_tag)
		h = dpg.get_value(self.roi_h_tag)
		
		# Calculate coords (Center logic)
		x = self.mouse_x - w // 2
		y = self.mouse_y - h // 2
		
		# Clip to image bounds
		img_w, img_h = self.texture_width, self.texture_height
		if img_w == 0 or img_h == 0: return # No image loaded yet

		x1 = max(0, x)
		y1 = max(0, y)
		x2 = min(img_w, x + w)
		y2 = min(img_h, y + h)

		if x2 > x1 and y2 > y1:
			dpg.configure_item(rect_tag, pmin=(x1, y1), pmax=(x2, y2), show=True)
			dpg.configure_item(text_tag, default_value=(x2, y2), show=True)
		else:
			dpg.configure_item(rect_tag, show=False)
			dpg.configure_item(text_tag, show=False)
			return

		# Calculate Stats on the fly if image data is available
		# This might be slightly heavy for main thread if image is huge, 
		# but for ROI it's usually small.
		if self.last_image is not None:
			try:
				# Coordinate conversion (Mirror process_roi)
				y1_arr = img_h - y2
				y2_arr = img_h - y1
				
				# Ensure valid slice
				y1_arr = max(0, min(y1_arr, img_h))
				y2_arr = max(0, min(y2_arr, img_h))
				
				roi = self.last_image[y1_arr:y2_arr, x1:x2]
				
				if roi.size > 0:
					thresh_pct = dpg.get_value(self.roi_thresh_tag)
					if roi.ndim == 3 and roi.shape[2] >= 3:
						mask_base = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY if roi.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
					else:
						mask_base = roi
						
					p_thresh = np.percentile(mask_base, 100 - thresh_pct)
					mask = mask_base >= p_thresh
					
					if roi.ndim == 3 and roi.shape[2] >= 3:
						valid_pixels = roi[mask]
						if roi.shape[2] == 3:
							valid_pixels_rgb = valid_pixels[:, ::-1] # BGR to RGB
						else:
							valid_pixels_rgb = valid_pixels[:, :3][:, ::-1] # BGRA to RGB
						mean_val = np.mean(valid_pixels_rgb, axis=0) if valid_pixels.size > 0 else [0.0, 0.0, 0.0]
						dpg.set_value(self.roi_result_tag, f"R:{mean_val[0]:.1f} G:{mean_val[1]:.1f} B:{mean_val[2]:.1f}")
						dpg.configure_item(text_tag, label=f"RGB: {mean_val[0]:.0f},{mean_val[1]:.0f},{mean_val[2]:.0f}")
					else:
						valid_pixels = mask_base[mask]
						mean_val = np.mean(valid_pixels) if valid_pixels.size > 0 else 0.0
						std_val = np.std(valid_pixels) if valid_pixels.size > 0 else 0.0
						
						dpg.set_value(self.roi_result_tag, f"{mean_val:.2f} , {std_val:.2f}")
						dpg.configure_item(text_tag, label=f"{mean_val:.1f} , {std_val:.1f}")
			except Exception:
				pass

	def process_roi(self, frame, processed_frame):
		"""Calculates ROI stats using raw frame and annotates processed_frame."""
		if not dpg.get_value(self.roi_enable_tag):
			return processed_frame

		# We still calculate mask for visualization if requested
		show_mask = dpg.get_value(self.roi_show_mask_tag)
		if not show_mask:
			return processed_frame

		# Get ROI parameters (We need to recalculate coords here for the thread)
		# NOTE: This runs in a thread, self.mouse_x/y might change during execution.
		# Ideally we snapshot them, but for visual mask it's okay.
		x = self.mouse_x
		y = self.mouse_y
		w = dpg.get_value(self.roi_w_tag)
		h = dpg.get_value(self.roi_h_tag)
		thresh_pct = dpg.get_value(self.roi_thresh_tag)

		# Center ROI on mouse
		x = x - w // 2
		y = y - h // 2

		# Ensure ROI is within bounds
		img_h, img_w = frame.shape[:2]
		
		# Clip coordinates to image
		x1 = max(0, x)
		y1 = max(0, y)
		x2 = min(img_w, x + w)
		y2 = min(img_h, y + h)
		
		# DPG Visualization is now handled in _update_roi_overlay (Main Thread)
		# We only handle mask stamping here.

		# Calculation (on Raw data)
		# Correct for coordinate system mismatch:
		# DPG Plot (0,0) is Bottom-Left. Image Array (0,0) is Top-Left.
		
		# Invert Y for array slicing
		y1_arr = img_h - y2
		y2_arr = img_h - y1
		
		# Ensure valid slice (handle negative indices from OOB)
		y1_arr = max(0, min(y1_arr, img_h))
		y2_arr = max(0, min(y2_arr, img_h))
		
		roi = frame[y1_arr:y2_arr, x1:x2]
		
		if roi.size > 0:
			# Calculate Threshold
			if roi.ndim == 3 and roi.shape[2] >= 3:
				mask_base = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY if roi.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
			else:
				mask_base = roi
				
			p_thresh = np.percentile(mask_base, 100 - thresh_pct)
			
			# Create mask
			mask = mask_base >= p_thresh
			
			# Show Selection Mask (Red overlay) - Still needs OpenCV modification of texture
			if show_mask:
				vis_roi = processed_frame[y1_arr:y2_arr, x1:x2]
				color = [0, 0, 255]
				if vis_roi.ndim == 3 and vis_roi.shape[2] == 4:
					color = [0, 0, 255, 255]
				vis_roi[mask] = color
				
		return processed_frame

	def process_lowdepth(self, frame, min_val=None, max_val=None, mask=None):
		"""Processes image before rendering and adds overlay if mask is provided."""
		frame = self.remap_and_annotate(frame, min_val, max_val)
		if mask is not None:
			frame[mask] = [0, 0, 255]
		return frame

	def _emit_cropped_frame(self, frame, force_emit=False):
		if not self.connections.get("Cropped Frame"):
			return

		if not force_emit and not dpg.get_value(self.roi_live_crop_tag):
			return

		img_h, img_w = frame.shape[:2]
		
		active_rect = None
		last_uuid = getattr(self, "last_selected_roi_uuid", None)
		
		if last_uuid and last_uuid in self.rois and self.rois[last_uuid].get("selected", False):
			active_rect = self.rois[last_uuid]["rect"]
		else:
			for r_uuid, r_data in reversed(list(self.rois.items())):
				if r_data.get("selected", False):
					active_rect = r_data["rect"]
					self.last_selected_roi_uuid = r_uuid
					break
				
		if active_rect is None and dpg.get_value(self.roi_enable_tag) and self.is_hovering:
			w = dpg.get_value(self.roi_w_tag)
			h = dpg.get_value(self.roi_h_tag)
			x = self.mouse_x - w // 2
			y = self.mouse_y - h // 2
			active_rect = [x, y, w, h]
			
		if active_rect:
			x, y, w, h = active_rect
			x1 = max(0, x)
			y1 = max(0, y)
			x2 = min(img_w, x + w)
			y2 = min(img_h, y + h)

			y1_arr = img_h - y2
			y2_arr = img_h - y1
			y1_arr = max(0, min(y1_arr, img_h))
			y2_arr = max(0, min(y2_arr, img_h))
			
			crop = frame[y1_arr:y2_arr, x1:x2]
			if crop.size > 0:
				for module in self.connections.get("Cropped Frame", []):
					module.input_cb(data=crop, data_type=IOTypes.FRAME)

	def _emit_lowdepth_frame(self, frame, force_emit=False):
		if not self.connections.get("Lowdepth Frame"):
			return

		minval, maxval = self._get_current_scale_limits()
		lowdepth = self.convert_to_lowdepth(frame)
		image = self.remap_and_annotate(lowdepth, minval, maxval)
		
		for module in self.connections.get("Lowdepth Frame", []):
			module.input_cb(data=image, data_type=IOTypes.FRAME)

	def input_cb(self, *args, **kwargs):
		"""
		Node editor input callback.
		Accepts frames, paths, samples, and (frame, mask) FRAME_MASK_PAIR.
		"""
		data = kwargs.get("data") if "data" in kwargs else (args[0] if args else None)
		data_type = kwargs.get("data_type", None)

		if data is None and "sample" in kwargs:
			data = kwargs["sample"]
		if data is None and "frame" in kwargs:
			data = kwargs["frame"]

		if data is None:
			return

		# Check for FRAME_MASK_PAIR: (frame, mask)
		is_frame_mask_pair = (
			data_type == IOTypes.FRAME_MASK_PAIR
			or (isinstance(data, (tuple, list)) and len(data) == 2 
				and isinstance(data[0], np.ndarray) and isinstance(data[1], np.ndarray))
		)

		if is_frame_mask_pair:
			raw_frame, mask = data[0], data[1]
			if self.use_rgba:
				# Apply mask to alpha channel: non-selected area (mask == 0) is transparent
				h, w = raw_frame.shape[:2]
				if mask.dtype == bool:
					alpha = np.where(mask, 255, 0).astype(np.uint8)
				elif mask.max() <= 1 and mask.dtype != np.uint8:
					alpha = (mask * 255).astype(np.uint8)
				else:
					alpha = mask.astype(np.uint8)
				
				if alpha.shape[:2] != (h, w):
					alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_NEAREST)

				if raw_frame.ndim == 2:
					bgr = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR)
				elif raw_frame.shape[2] == 4:
					bgr = raw_frame[:, :, :3]
				else:
					bgr = raw_frame
				
				frame = cv2.merge([bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2], alpha])
			else:
				frame = raw_frame
			self.update_image(frame)
			return

		if isinstance(data, (tuple, list)) and len(data) >= 1 and isinstance(data[0], np.ndarray):
			data = data[0]

		if isinstance(data, np.ndarray):
			if self.use_rgba and data_type == IOTypes.MASK:
				mask = data
				alpha = np.where(mask > 0, 255, 0).astype(np.uint8) if mask.dtype != np.uint8 else mask
				frame = cv2.merge([alpha, alpha, alpha, alpha])
			else:
				frame = data
			self.update_image(frame)
		elif isinstance(data, str) and os.path.exists(data):
			img = cv2.imread(data, cv2.IMREAD_UNCHANGED)
			if img is not None:
				self.update_image(img)

	def update_image(self, frame):
		"""
		Receives a new frame, emits dependent frames, and signals render thread (non-blocking).
		"""
		if frame is None:
			return
		
		# Calculate incoming FPS
		self.calc_fps()
		
		self._emit_cropped_frame(frame)
		self._emit_lowdepth_frame(frame)
		
		self.last_image = frame
		with self.frame_lock:
			self.latest_frame = frame
		self.update_event.set()

	def _render_loop(self):
		"""Internal render loop that displays the latest available frame without blocking input."""
		while self.running:
			self.update_event.wait(timeout=0.1)
			self.update_event.clear()

			frame_to_render = None
			with self.frame_lock:
				if self.latest_frame is not None:
					frame_to_render = self.latest_frame
					self.latest_frame = None

			if frame_to_render is not None:
				try:
					self._render_frame(frame_to_render)
				except Exception as e:
					logger.warning(f"Error in Image Viewer render loop: {e}")

	def update_image_callback(self, *args, **kwargs):
		"""Callback to trigger update from UI."""
		if self.last_image is not None:
			with self.frame_lock:
				self.latest_frame = self.last_image
			self.update_event.set()

		# Also update overlay immediately for responsiveness (lightweight)
		self._update_roi_overlay()

	def _render_frame(self, frame):
		"""Processes the frame and updates the GPU texture."""
		if frame is None:
			return
		with self.lock:
			try:
				# Resize if needed (main-thread safe)
				h, w = frame.shape[:2]
				if h != self.texture_height or w != self.texture_width:
					self.init_viewer(w, h)
				
				# Handle color images (3 or 4 channels) vs grayscale
				if frame.ndim == 3 and frame.shape[2] in [3, 4] and frame.dtype == np.uint8:
					processed = frame.copy()
					processed = self.process_roi(frame, processed)
				else:
					lowdepth = self.convert_to_lowdepth(frame)
					minval, maxval = self._get_current_scale_limits()
					processed = self.process_lowdepth(lowdepth, minval, maxval)
					processed = self.process_roi(frame, processed)
				
				# Update Texture
				dpg.set_value(self.texture_tag, self.convert_to_texture(processed))
				
				# FPS Display Update
				if self._frame_count % self._fps_update_interval == 0:
					if dpg.does_item_exist(self.fps_tag):
						dpg.set_value(self.fps_tag, str(int(self.fps)))
				self._frame_count += 1
				
			except Exception as e:
				logger.error(f"Error in Image Viewer render: {e}")

	def set_mode(self, sender, app_data, user_data=None, *args):
		"""Handles UI mode changes (negative toggle or palette switch)."""
		if app_data in ["Normal", "Negative"]:
			self.negative = (app_data == "Negative")
		else:
			self.palette = app_data
		
		if self.last_image is not None:
			self._render_frame(self.last_image)

	def set_depth(self, sender, app_data, user_data=None, *args):
		"""Handles image depth changes."""
		self.image_depth = app_data
		if self.last_image is not None:
			self._render_frame(self.last_image)
		self._update_roi_overlay()

	def copy_to_clipboard(self, *args, **kwargs):
		"""Copies the processed image to clipboard using the custom injector."""
		if self.last_image is not None:
			minval, maxval = self._get_current_scale_limits()
			lowdepth = self.convert_to_lowdepth(self.last_image)
			image = self.remap_and_annotate(lowdepth, minval, maxval)
			clipboardinjector.send_image(image)

	def save_image(self, *args, **kwargs):
		"""Saves both raw and processed images to PNG with indexed names."""
		if self.last_image is not None:
			minval, maxval = self._get_current_scale_limits()
			raw = self.last_image
			lowdepth = self.convert_to_lowdepth(raw)
			processed = self.remap_and_annotate(lowdepth, minval, maxval)

			name = dpg.get_value(self.name_tag)
			
			if self.export_path:
				os.makedirs(self.export_path, exist_ok=True)
				raw_path = os.path.join(self.export_path, f"{name}_RAW_{self.imgcount}.png")
				processed_path = os.path.join(self.export_path, f"{name}_Processed_{self.imgcount}.png")
			else:
				raw_path = f"{name}_RAW_{self.imgcount}.png"
				processed_path = f"{name}_Processed_{self.imgcount}.png"

			cv2.imwrite(raw_path, raw)
			cv2.imwrite(processed_path, processed)

			dpg.set_value(self.count_tag, f"count : {self.imgcount}")
			self.imgcount += 1

	def update_save_tooltip(self):
		"""Updates the tooltip for the Save button based on the current export path."""
		if hasattr(self, 'save_tooltip_text_tag') and dpg.does_item_exist(self.save_tooltip_text_tag):
			tip_text = f"Saves to: {self.export_path}" if self.export_path else "Saves to default folder"
			dpg.set_value(self.save_tooltip_text_tag, tip_text)

	def reset_counter(self, *args, **kwargs):
		"""Resets the saved image counter to zero."""
		self.imgcount = 0
		dpg.set_value(self.count_tag, f"count: {self.imgcount}")

	def _update_categories_count(self, sender, app_data, user_data=None):
		self.categories_count = app_data

	def send_palette_to_clipboard(self, *args, **kwargs):
		"""Computes and sends the current category palette to the clipboard in palette.json format."""
		if self.last_image is None:
			logger.warning("No image loaded to compute palette.")
			return
		
		# Get number of categories from UI
		num_categories = dpg.get_value(self.cat_count_tag)
		if num_categories < 1:
			logger.warning("Number of categories must be at least 1.")
			return

		minval, maxval = self._get_current_scale_limits()
		range_diff = maxval - minval
		if range_diff <= 0:
			range_diff = 1.0

		palette_dict = {}
		for i in range(num_categories):
			bin_low = minval + i * range_diff / num_categories
			bin_high = minval + (i + 1) * range_diff / num_categories
			cat_name = f"{bin_low/1000.0:.3f}-{bin_high/1000.0:.3f}"
			
			val_center = int((i + 0.5) * 255 / num_categories)
			pixel_img = np.array([[val_center]], dtype=np.uint8)
			colorized = self.colorize_frame(pixel_img)
			if colorized.ndim == 3 and colorized.shape[2] >= 3:
				bgr = colorized[0, 0]
				rgb = [int(bgr[2]), int(bgr[1]), int(bgr[0])]
			else:
				val = int(colorized[0, 0] if colorized.ndim >= 2 else val_center)
				rgb = [val, val, val]
			palette_dict[cat_name] = rgb

		import json
		palette_json = json.dumps(palette_dict, indent=4)
		clipboardinjector.send_text(palette_json)
		logger.success(f"Palette with {num_categories} categories copied to clipboard in palette.json format.")

	def send_result_to_clipboard(self, *args, **kwargs):
		"""Computes the category distribution and sends it to the clipboard in TSV format."""
		if self.last_image is None:
			logger.warning("No image loaded to compute proportions.")
			return

		num_categories = dpg.get_value(self.cat_count_tag)
		if num_categories < 1:
			logger.warning("Number of categories must be at least 1.")
			return

		# Convert raw image to a single channel intensity map for thresholding and binning
		if self.last_image.ndim == 3 and self.last_image.shape[2] >= 3:
			raw_gray = cv2.cvtColor(self.last_image, cv2.COLOR_BGR2GRAY if self.last_image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
		else:
			raw_gray = self.last_image

		# Get current scale limits in raw values
		minval, maxval = self._get_current_scale_limits()

		# Determine the region of interest
		img_h, img_w = raw_gray.shape[:2]
		active_rect = None
		last_uuid = getattr(self, "last_selected_roi_uuid", None)
		thresh_pct = 100
		
		if last_uuid and last_uuid in self.rois and self.rois[last_uuid].get("selected", False):
			active_rect = self.rois[last_uuid]["rect"]
			thresh_pct = self.rois[last_uuid].get("threshold", 100)
		else:
			for r_uuid, r_data in reversed(list(self.rois.items())):
				if r_data.get("selected", False):
					active_rect = r_data["rect"]
					self.last_selected_roi_uuid = r_uuid
					thresh_pct = r_data.get("threshold", 100)
					break

		if active_rect:
			x, y, w, h = active_rect
			x1 = max(0, x)
			y1 = max(0, y)
			x2 = min(img_w, x + w)
			y2 = min(img_h, y + h)

			# Invert Y coordinate for numpy array slicing (coordinate system mismatch)
			y1_arr = img_h - y2
			y2_arr = img_h - y1
			y1_arr = max(0, min(y1_arr, img_h))
			y2_arr = max(0, min(y2_arr, img_h))

			roi_raw = raw_gray[y1_arr:y2_arr, x1:x2]
			
			if roi_raw.size > 0:
				p_thresh = np.percentile(roi_raw, 100 - thresh_pct)
				# Exclude zero values
				mask = (roi_raw >= p_thresh) & (roi_raw > 0)
				target_raw = roi_raw[mask]
			else:
				target_raw = np.array([], dtype=raw_gray.dtype)
		else:
			# Exclude zero values from the entire image
			mask = raw_gray > 0
			target_raw = raw_gray[mask]

		# Compute range and binning
		range_diff = maxval - minval
		if range_diff <= 0:
			range_diff = 1.0

		if target_raw.size > 0:
			# Normalize raw intensities to [0.0, 1.0] range
			normalized = (target_raw.astype(np.float32) - minval) / range_diff
			np.clip(normalized, 0.0, 1.0, out=normalized)
			
			# Map to [0, num_categories - 1] bin indices
			indices = (normalized * num_categories).astype(np.int32)
			indices = np.clip(indices, 0, num_categories - 1)
			
			bin_counts = np.bincount(indices, minlength=num_categories)
		else:
			bin_counts = np.zeros(num_categories, dtype=np.int32)

		total_pixels = np.sum(bin_counts)
		lines = ["Label\tCount\tPercentage (%)"]
		for i in range(num_categories):
			bin_low = minval + i * range_diff / num_categories
			bin_high = minval + (i + 1) * range_diff / num_categories
			cat_name = f"{bin_low/1000.0:.3f}-{bin_high/1000.0:.3f}"
			count = int(bin_counts[i])
			percent = (100.0 * count / total_pixels) if total_pixels > 0 else 0.0
			lines.append(f"{cat_name}\t{count}\t{percent:.2f}")

		tsv_data = "\n".join(lines)
		clipboardinjector.send_text(tsv_data)
		logger.success(f"Category distribution copied to clipboard in TSV format.")

	def on_close(self):
		"""Release resources."""
		self.running = False
		self.update_event.set()
		if dpg.does_item_exist(self.texture_reg_tag):
			dpg.delete_item(self.texture_reg_tag)
		if hasattr(self, "worker_thread") and self.worker_thread.is_alive():
			self.worker_thread.join(timeout=0.2)


EXPORTED_CLASS = Image_viewer_win
EXPORTED_NAME = "Image Viewer"
