import dearpygui.dearpygui as dpg
from core.window_base import WindowBase
from core.input_output_types import IOTypes
from loguru import logger
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import cv2
from modules.plot.clipboard_injector import clipboardinjector

class ColorPieWin(WindowBase):
    """
    Specialized Matplotlib module for rendering Circular (Pie) plots of color distributions.
    Receives color statistics from ColorMapper or equivalent modules.
    """
    
    def __init__(self,
                label="Color Distribution Plot",
                win_width=600,
                win_height=500,
                pos=(50, 450),
                uuid=None,
                outputs=None,
                visible=True):
        
        outputs = outputs or {
            "Frame": IOTypes.FRAME
        }
            
        super().__init__(
            label=label,
            pos=pos,
            win_width=win_width,
            win_height=win_height,
            uuid=uuid,
            outputs=outputs,
            visible=visible
        )
        
        self.accepted_input_types = [IOTypes.CMD_DICT]
        
        # State
        self.canvas_width = 500
        self.canvas_height = 400
        self.dpi = 100
        self.last_bgr_frame = None
        self._last_data = None
        
        # Texture Handling
        self.texture_registry_tag = f"pie_registry_{self.UUID}"
        self.texture_tag = f"pie_texture_{self.UUID}"
        
        with dpg.texture_registry(tag=self.texture_registry_tag, show=False):
            dpg.add_raw_texture(
                width=self.canvas_width, 
                height=self.canvas_height, 
                default_value=np.zeros(self.canvas_width * self.canvas_height * 3, dtype=np.float32),
                tag=self.texture_tag, 
                format=dpg.mvFormat_Float_rgb
            )
        
        self._build_ui()

        self.update_permission()

    def update_permission(self):
        """
        Adjust module permissions and UI visibility based on app_state.mode.
        Disable closing in user mode.
        """
        from core.app_state import app_state
        mode = app_state.mode
        
        if dpg.does_item_exist(self.winID):
            is_user = (mode == "user")
            dpg.configure_item(self.winID, no_close=is_user)

    def _build_ui(self):
        with dpg.window(label=self.label, width=self.win_width, height=self.win_height, pos=self.pos, tag=self.winID, show=self.visible):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Copy Image", callback=self.copy_to_clipboard)
                dpg.add_button(label="Export TSV (Excel)", callback=self.export_tsv_to_clipboard)
                dpg.add_button(label="Save", callback=self._save_frame_trigger)
            
            dpg.add_separator()
            
            # Simple Plot Area
            with dpg.plot(height=-1, width=-1, no_menus=True, tag=f"pie_plot_{self.UUID}"):
                dpg.add_plot_axis(dpg.mvXAxis, no_gridlines=True, no_tick_labels=True, no_tick_marks=True)
                with dpg.plot_axis(dpg.mvYAxis, no_gridlines=True, no_tick_labels=True, no_tick_marks=True):
                    dpg.add_image_series(self.texture_tag, [0, 0], [self.canvas_width, self.canvas_height])

    def input_cb(self, data, *args, **kwargs):
        if not isinstance(data, dict):
            return
            
        if data.get("action") == "plot_pie":
            self._last_data = data
            self._render_pie(data)

    def _render_pie(self, data):
        try:
            dist = data.get("distribution", {})
            if not dist:
                return
            
            # Filter out non-dictionary metadata (like 'best_match')
            plot_dist = {k: v for k, v in dist.items() if isinstance(v, dict) and "count" in v}
            if not plot_dist:
                return
            
            # Prepare Labels/Counts/Colors
            labels = list(plot_dist.keys())
            counts = [v["count"] for v in plot_dist.values()]
            colors_rgb = [np.array(v["color"])/255.0 for v in plot_dist.values()]
            
            # Total for percentages
            total = sum(counts)
            
            # Matplotlib Render
            fig = Figure(figsize=(self.canvas_width/self.dpi, self.canvas_height/self.dpi), dpi=self.dpi, facecolor='#1e1e1e')
            canvas = FigureCanvas(fig)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_facecolor('#1e1e1e')
            
            # Use small padding for pie to look premium
            wedges, texts, autotexts = ax.pie(
                counts, 
                labels=labels, 
                autopct='%1.1f%%', 
                colors=colors_rgb,
                startangle=140,
                textprops={'color': "w", 'fontsize': 10},
                wedgeprops={'edgecolor': 'gray', 'linewidth': 0.5, 'antialiased': True}
            )
            
            # Cleanly render as numpy array
            canvas.draw()
            rgba_buffer = canvas.buffer_rgba()
            img_rgba = np.asarray(rgba_buffer)
            
            # Convert to BGR for Node Frame output
            img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
            self.last_bgr_frame = img_bgr
            
            # UI Texture update (RGB)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            flattened = img_rgb.ravel().astype(np.float32) / 255.0
            
            if flattened.size == (self.canvas_width * self.canvas_height * 3):
                dpg.set_value(self.texture_tag, flattened)
            
            # Emit Frame
            for module in self.connections.get("Frame", []):
                module.input_cb(data=img_bgr, data_type=IOTypes.FRAME)
                
        except Exception as e:
            logger.error(f"Pie Plot Error: {e}")

    def copy_to_clipboard(self, *args, **kwargs):
        if self.last_bgr_frame is not None:
            clipboardinjector.send_image(self.last_bgr_frame)
            logger.success("Pie plot image copied to clipboard.")

    def export_tsv_to_clipboard(self, *args, **kwargs):
        if self._last_data is None:
            logger.warning("No data to export.")
            return
            
        dist = self._last_data.get("distribution", {})
        if not dist:
            logger.warning("Distribution data is empty.")
            return

        # Build TSV String
        lines = ["Label\tCount\tPercentage (%)"]
        for label, values in dist.items():
            if not isinstance(values, dict):
                continue
            count = values.get("count", 0)
            percent = values.get("percent", 0.0)
            lines.append(f"{label}\t{count}\t{percent:.2f}")
            
        tsv_data = "\n".join(lines)
        clipboardinjector.send_text(tsv_data)
        logger.success(f"TSV data ({len(dist)} lines) copied to clipboard. Ready for Excel.")

    def _save_frame_trigger(self, *args, **kwargs):
        # Place holder for save logic if needed
        logger.info("Save to file requested (not implemented).")

EXPORTED_CLASS = ColorPieWin
EXPORTED_NAME = "Color Pie Plot"
