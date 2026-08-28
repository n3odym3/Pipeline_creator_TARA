import dearpygui.dearpygui as dpg
from core.window_base import WindowBase
from core.input_output_types import IOTypes
from loguru import logger
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import cv2
import seaborn as sns
from modules.plot.clipboard_injector import clipboardinjector

class Matplot_win(WindowBase):
    """
    Window module for displaying Matplotlib plots directly in Node-assistant.
    Receives data from Lineplot and renders it into a DPG texture.
    """
    
    def __init__(self,
                label="Matplotlib Viewer",
                win_width=800,
                win_height=600,
                pos=(50, 50),
                uuid=None,
                outputs=None,
                visible=True,
                plot_title="",
                xlabel="",
                ylabel="",
                dpi=100,
                canvas_width=800,
                canvas_height=600,
                show_legend=True,
                use_seaborn=True,
                use_shading=False):
        
        if outputs is None:
            outputs = {
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
        self.connections = {k: [] for k in self.outputs}
        
        # Plot Configuration State
        self.plot_title = plot_title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.dpi = dpi
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.show_legend = show_legend
        self.use_seaborn = use_seaborn
        self.use_shading = use_shading
        
        self._persistent_fields = ["plot_title", "xlabel", "ylabel", "dpi", "canvas_width", "canvas_height", "show_legend", "use_seaborn", "use_shading"]
        
        # Texture Handling
        self.texture_registry_tag = f"matplot_registry_{self.UUID}"
        self.texture_tag = f"matplot_texture_{self.UUID}"
        self.texture_width = self.canvas_width
        self.texture_height = self.canvas_height
        self.last_bgr_frame = None
        self.last_plot_data = None # Store most recent incoming data
        
        # Create persistent registry
        with dpg.texture_registry(tag=self.texture_registry_tag, show=False):
            dpg.add_raw_texture(
                width=self.texture_width, 
                height=self.texture_height, 
                default_value=np.zeros(self.texture_width * self.texture_height * 3, dtype=np.float32),
                tag=self.texture_tag, 
                format=dpg.mvFormat_Float_rgb
            )
        
        with dpg.window(
            label=self.label,
            width=self.win_width,
            height=self.win_height,
            pos=self.pos,
            tag=self.winID,
            show=self.visible
        ):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Clipboard", callback=self.copy_to_clipboard)
                dpg.add_button(label="Advanced Options", callback=self._show_advanced_options)
            
            dpg.add_separator()
            # Plot wrapper to easily recreate the plot on resize
            with dpg.group(tag=f"matplot_wrapper_{self.UUID}"):
                self._build_plot()

    def _show_advanced_options(self, *args, **kwargs):
        config_win = f"matplot_config_{self.UUID}"
        if dpg.does_item_exist(config_win):
            dpg.show_item(config_win)
            dpg.focus_item(config_win)
            return

        with dpg.window(label=f"Config - {self.label}", tag=config_win, width=300, height=350, pos=(self.pos[0]+50, self.pos[1]+50)):
            dpg.add_input_text(label="Title", default_value=self.plot_title, callback=lambda s, a: setattr(self, "plot_title", a))
            dpg.add_input_text(label="X Label", default_value=self.xlabel, callback=lambda s, a: setattr(self, "xlabel", a))
            dpg.add_input_text(label="Y Label", default_value=self.ylabel, callback=lambda s, a: setattr(self, "ylabel", a))
            dpg.add_separator()
            dpg.add_input_int(label="Width", default_value=self.canvas_width, callback=lambda s, a: setattr(self, "canvas_width", a))
            dpg.add_input_int(label="Height", default_value=self.canvas_height, callback=lambda s, a: setattr(self, "canvas_height", a))
            dpg.add_input_int(label="DPI", default_value=self.dpi, callback=lambda s, a: setattr(self, "dpi", a))
            dpg.add_checkbox(label="Show Legend", default_value=self.show_legend, callback=lambda s, a: setattr(self, "show_legend", a))
            dpg.add_checkbox(label="Seaborn Look", default_value=self.use_seaborn, callback=lambda s, a: setattr(self, "use_seaborn", a))
            dpg.add_checkbox(label="Shading", default_value=self.use_shading, callback=lambda s, a: setattr(self, "use_shading", a))
            dpg.add_separator()
            dpg.add_button(label="Apply and Refresh", width=-1, callback=self._apply_config)

    def _build_plot(self):
        """Helper to build or rebuild the plot."""
        with dpg.plot(height=-1, width=-1, no_menus=True, tag=f"matplot_plot_{self.UUID}", parent=f"matplot_wrapper_{self.UUID}"):
            xaxis = dpg.add_plot_axis(dpg.mvXAxis, no_gridlines=True, no_tick_labels=True, no_tick_marks=True)
            with dpg.plot_axis(dpg.mvYAxis, no_gridlines=True, no_tick_labels=True, no_tick_marks=True) as yaxis:
                dpg.add_image_series(self.texture_tag, [0, 0], [self.texture_width, self.texture_height])
                dpg.fit_axis_data(xaxis)
                dpg.fit_axis_data(yaxis)

    def _apply_config(self, *args, **kwargs):
        # Update internal texture if size changed
        if self.canvas_width != self.texture_width or self.canvas_height != self.texture_height:
            self.texture_width = self.canvas_width
            self.texture_height = self.canvas_height
            
            # 1. Cleanup old items
            if dpg.does_item_exist(self.texture_tag):
                dpg.delete_item(self.texture_tag)
            if dpg.does_item_exist(f"matplot_plot_{self.UUID}"):
                dpg.delete_item(f"matplot_plot_{self.UUID}")
            
            # 2. Recreate texture in persistent registry
            dpg.add_raw_texture(
                width=self.texture_width, 
                height=self.texture_height, 
                default_value=np.zeros(self.texture_width * self.texture_height * 3, dtype=np.float32),
                tag=self.texture_tag, 
                format=dpg.mvFormat_Float_rgb,
                parent=self.texture_registry_tag
            )
            
            # 3. Rebuild plot with new bounds
            self._build_plot()
        
        if self.last_plot_data:
            self.render_plot(self.last_plot_data)

    def copy_to_clipboard(self, *args, **kwargs):
        """Copies the latest plot image to the clipboard."""
        if self.last_bgr_frame is not None:
            clipboardinjector.send_image(self.last_bgr_frame)
            logger.success("Plot copied to clipboard.")
        else:
            logger.warning("No plot to copy.")

    def input_cb(self, data=None, *args, **kwargs):
        """Handle incoming command dictionary."""
        cmd = None
        if isinstance(data, dict):
            cmd = data
        elif kwargs.get("data_type") == IOTypes.CMD_DICT:
            cmd = kwargs.get("data")
            
        if cmd and cmd.get("action") == "plot_data":
            self.last_plot_data = cmd
            dpg.show_item(self.winID)
            dpg.focus_item(self.winID)
            self.render_plot(cmd)

    def render_plot(self, plot_data):
        """Creates the Matplotlib plot and updates the DPG texture."""
        try:
            dpi = self.dpi
            fig = Figure(figsize=(self.texture_width/dpi, self.texture_height/dpi), dpi=dpi)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            
            # Apply Seaborn style if enabled
            if self.use_seaborn:
                sns.set_theme(style="darkgrid", palette="muted")
                # When using sns.set_theme, it affects global rcParams, 
                # but fig.add_subplot will pick them up.
            else:
                # Reset to default or standard plt style
                plt.rcParams.update(plt.rcParamsDefault)
                ax.grid(True)

            series = plot_data.get("series", [])
            for s in series:
                x_vals = np.array(s["x"])
                y_vals = np.array(s["y"])
                
                # Check for duplicates in X
                unique_x = np.unique(x_vals)
                
                if len(unique_x) < len(x_vals):
                    # Duplicate X values found: Aggregate
                    means = []
                    stds = []
                    for ux in unique_x:
                        mask = x_vals == ux
                        points = y_vals[mask]
                        means.append(np.mean(points))
                        stds.append(np.std(points))
                    
                    means = np.array(means)
                    stds = np.array(stds)
                    
                    if self.use_seaborn:
                        sns.lineplot(x=x_vals, y=y_vals, label=s["label"], ax=ax, errorbar='sd', err_style="band")
                        color = ax.lines[-1].get_color()
                    else:
                        line, = ax.plot(unique_x, means, label=s["label"])
                        color = line.get_color()
                        ax.fill_between(unique_x, means - stds, means + stds, color=color, alpha=0.2)
                    
                    if self.use_shading:
                        ax.fill_between(unique_x, means, color=color, alpha=0.15)
                else:
                    # Standard plot
                    if self.use_seaborn:
                        sns.lineplot(x=s["x"], y=s["y"], label=s["label"], ax=ax)
                        color = ax.lines[-1].get_color()
                    else:
                        line, = ax.plot(s["x"], s["y"], label=s["label"])
                        color = line.get_color()
                        
                    if self.use_shading:
                        ax.fill_between(s["x"], s["y"], color=color, alpha=0.15)
                
            ax.set_title(self.plot_title if self.plot_title else plot_data.get("title", "Plot"))
            ax.set_xlabel(self.xlabel if self.xlabel else plot_data.get("xlabel", "X"))
            ax.set_ylabel(self.ylabel if self.ylabel else plot_data.get("ylabel", "Y"))
            
            # Apply synced limits if provided
            x_lim = plot_data.get("x_lim")
            y_lim = plot_data.get("y_lim")
            if x_lim:
                ax.set_xlim(x_lim)
            if y_lim:
                ax.set_ylim(y_lim)

            if plot_data.get("log_x"):
                ax.set_xscale('log')

            if self.show_legend:
                ax.legend()
            ax.grid(True)
            
            # Efficient rendering to numpy array
            canvas.draw()
            rgba_buffer = canvas.buffer_rgba()
            img_rgba = np.asarray(rgba_buffer)
            
            # Matplotlib provides RGBA. We need BGR for the Frame output (standard in Node-assistant)
            img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
            self.last_bgr_frame = img_bgr
            
            # Emit Frame (BGR)
            self._emit_frame(img_bgr)

            # For internal DPG texture, we use RGB (like Image Viewer)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Update DPG Texture (convert to flattened float-RGB)
            flattened = img_rgb.ravel().astype(np.float32) / 255.0
            dpg.set_value(self.texture_tag, flattened)
            
            logger.success("Matplotlib plot rendered & emitted (BGR).")
            
        except Exception as e:
            logger.error(f"Failed to render Matplotlib plot to DPG: {e}")

    def _emit_frame(self, frame):
        for module in self.connections.get("Frame", []):
            if hasattr(module, 'input_cb'):
                module.input_cb(data=frame, data_type=IOTypes.FRAME)

EXPORTED_CLASS = Matplot_win
EXPORTED_NAME = "Matplotlib Viewer"
