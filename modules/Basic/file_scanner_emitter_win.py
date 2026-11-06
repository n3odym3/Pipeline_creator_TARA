from __future__ import annotations

import fnmatch
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import dearpygui.dearpygui as dpg
from loguru import logger

from config.display_scaling import display_scaling
from core.file_explorer import file_explorer
from core.input_output_types import IOTypes
from core.window_base import WindowBase


class FileScannerEmitter_win(WindowBase):
    """
    File Scanner Emitter module for DearPyGui Pipeline Creator.
    Allows browsing a folder, recursively scanning for files matching a custom pattern/extension,
    and emitting file paths sequentially one-by-one upon receiving completion triggers.
    """

    label: str = "File Scanner Emitter"
    win_width: int = display_scaling.scale(360)
    win_height: int = display_scaling.scale(340)
    description: str = (
        "Recursively scan folders for matching files and emit paths sequentially on trigger."
    )

    accepted_input_types: list[str] = [
        IOTypes.FOLDER_PATH,
        IOTypes.FILE_PATH,
        IOTypes.TRIGGER,
        IOTypes.CMD_DICT,
    ]
    outputs: dict[str, str] = {
        "File Path": IOTypes.FILE_PATH,
        "Trigger": IOTypes.TRIGGER,
    }
    persistent_fields: list[str] = ["label", "folder", "pattern", "recursive"]

    def __init__(
        self,
        folder: str = "",
        pattern: str = "*.csv",
        recursive: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the File Scanner Emitter window module.

        Args:
            folder: Initial root directory to scan.
            pattern: File matching glob/extension pattern.
            recursive: Whether to scan subdirectories recursively.
            **kwargs: Additional parameters forwarded to WindowBase.
        """
        super().__init__(**kwargs)

        self.folder: str = getattr(self, "folder", None) or folder
        self.pattern: str = getattr(self, "pattern", None) or pattern
        self.recursive: bool = (
            getattr(self, "recursive", None)
            if getattr(self, "recursive", None) is not None
            else recursive
        )

        # Internal state
        self._tag_counter: int = 0
        self.file_tree: Optional[Dict[str, Any]] = None
        self.file_list: List[Dict[str, Any]] = []
        self.file_ui_map: Dict[str, Dict[str, Any]] = {}
        self.file_path_to_tag: Dict[str, str] = {}
        self.folder_map: Dict[str, Dict[str, Any]] = {}

        # Queue / Trigger state
        self.process_lock: threading.Lock = threading.Lock()
        self.process_queue: List[Dict[str, Any]] = []
        self.process_queue_index: int = 0
        self.is_queue_active: bool = False
        self.current_processing_path: str = ""
        self.last_selectable: Optional[str] = None

        # Tags
        uid = self.UUID
        self.folder_input_tag: str = f"fse_folder_{uid}"
        self.pattern_input_tag: str = f"fse_pattern_{uid}"
        self.recursive_chk_tag: str = f"fse_rec_{uid}"
        self.status_text_tag: str = f"fse_status_{uid}"
        self.send_all_btn_tag: str = f"fse_send_all_{uid}"
        self.stop_btn_tag: str = f"fse_stop_{uid}"
        self.tree_group_tag: str = f"fse_tree_group_{uid}"
        self.child_win_tree_tag: str = f"fse_child_tree_{uid}"

        # Color themes for visual states
        with dpg.theme() as self.theme_processing_blue:
            with dpg.theme_component(dpg.mvSelectable):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (100, 170, 255, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Header, (40, 80, 160, 255))

        with dpg.theme() as self.theme_processed_green:
            with dpg.theme_component(dpg.mvSelectable):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (100, 255, 120, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Header, (30, 120, 60, 255))

        self._build_ui()
        self.update_permission()

        if self.folder and os.path.isdir(self.folder):
            self.scan_and_refresh()

    def _build_ui(self) -> None:
        """Construct DearPyGui widgets for file scanner emitter."""
        scale = display_scaling.scale

        with self.window():
            # Row 1: Folder selection
            with dpg.group(horizontal=True):
                dpg.add_button(label="\uf07c Browse", callback=self._on_browse_clicked)
                dpg.add_input_text(
                    tag=self.folder_input_tag,
                    default_value=self.folder,
                    hint="Enter folder path or browse...",
                    width=-1,
                    callback=self._on_folder_input_changed,
                )

            # Row 2: Pattern filter & recursive
            with dpg.group(horizontal=True):
                dpg.add_text("Match:", color=(180, 180, 180))
                dpg.add_input_text(
                    tag=self.pattern_input_tag,
                    default_value=self.pattern,
                    hint="e.g. *.csv, _radial_detections.csv, *",
                    width=scale(160),
                    callback=self._on_pattern_changed,
                )
                dpg.add_checkbox(
                    label="Recursive",
                    tag=self.recursive_chk_tag,
                    default_value=self.recursive,
                    callback=self._on_recursive_changed,
                )
                dpg.add_button(label="\uf021 Scan", callback=self._on_scan_clicked)

            dpg.add_separator()

            # Row 3: Action controls
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="\uf04b Send ALL",
                    tag=self.send_all_btn_tag,
                    callback=self._on_send_all_clicked,
                )
                dpg.add_button(
                    label="\uf04d Stop",
                    tag=self.stop_btn_tag,
                    enabled=False,
                    callback=self._on_stop_clicked,
                )
                dpg.add_button(
                    label="\uf0e2 Reset Status",
                    callback=self._on_reset_status_clicked,
                )

            # Status row
            dpg.add_text("Ready - Select a folder to scan.", tag=self.status_text_tag, color=(160, 200, 255))
            dpg.add_separator()

            # Tree / File container
            with dpg.group(tag=self.tree_group_tag):
                with dpg.child_window(tag=self.child_win_tree_tag, border=True):
                    dpg.add_text("No folder selected.", color=(160, 160, 160))

    def update_permission(self) -> None:
        """Adjust module permissions and UI elements based on the application mode."""
        from core.app_state import app_state

        is_user = getattr(app_state, "mode", "dev") == "user"
        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def _next_id(self) -> int:
        self._tag_counter += 1
        return self._tag_counter

    def _on_browse_clicked(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Handle browse button click to select folder."""
        default_dir = self.folder if self.folder and os.path.isdir(self.folder) else ""
        file_explorer.select_folder(
            default_path=default_dir,
            callback=self._on_folder_selected,
        )

    def _on_folder_selected(self, path: str, *args: Any, **kwargs: Any) -> None:
        """Handle folder path returned from file explorer dialog."""
        if path and os.path.isdir(path):
            self.folder = path
            if dpg.does_item_exist(self.folder_input_tag):
                dpg.set_value(self.folder_input_tag, path)
            self.scan_and_refresh()

    def _on_folder_input_changed(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Handle manual text change in folder input."""
        self.folder = str(app_data or "").strip()
        if os.path.isdir(self.folder):
            self.scan_and_refresh()

    def _on_pattern_changed(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Handle pattern text input change."""
        self.pattern = str(app_data or "").strip()
        if self.folder and os.path.isdir(self.folder):
            self.scan_and_refresh()

    def _on_recursive_changed(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Handle recursive checkbox toggle."""
        self.recursive = bool(app_data)
        if self.folder and os.path.isdir(self.folder):
            self.scan_and_refresh()

    def _on_scan_clicked(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Trigger scan and refresh."""
        self.scan_and_refresh()

    def _matches_pattern(self, filename: str) -> bool:
        """Check if a filename matches the active glob or substring pattern."""
        pat = self.pattern.strip()
        if not pat or pat in ("*", "*.*"):
            return True

        fn_lower = filename.lower()
        pat_lower = pat.lower()

        if fnmatch.fnmatch(fn_lower, pat_lower):
            return True

        if pat_lower in fn_lower:
            return True

        if pat_lower.startswith(".") and fn_lower.endswith(pat_lower):
            return True

        return False

    def scan_directory(self, root_dir: str, recursive: bool = True) -> Optional[Dict[str, Any]]:
        """Scans root_dir and builds a tree of matching files."""
        if not root_dir or not os.path.isdir(root_dir):
            return None

        self.folder_map = {}

        def scan_node(current_path: str) -> Optional[Dict[str, Any]]:
            try:
                entries = os.listdir(current_path)
            except Exception as e:
                logger.warning(f"[{self.label}] Cannot access {current_path}: {e}")
                return None

            name = os.path.basename(current_path) or current_path
            subdirs: list[dict[str, Any]] = []
            files: list[dict[str, Any]] = []

            for entry in entries:
                full_path = os.path.join(current_path, entry)
                if os.path.isdir(full_path):
                    if recursive:
                        subdir_node = scan_node(full_path)
                        if subdir_node:
                            subdirs.append(subdir_node)
                else:
                    if self._matches_pattern(entry):
                        file_info = {
                            "name": entry,
                            "path": full_path,
                            "basename": os.path.splitext(entry)[0],
                            "processed": False,
                        }
                        files.append(file_info)

            subdirs.sort(key=lambda x: str(x["name"]).lower())
            files.sort(key=lambda x: str(x["name"]).lower())

            node = {
                "type": "folder",
                "name": name,
                "path": current_path,
                "subdirs": subdirs,
                "files": files,
            }
            self.folder_map[current_path] = node
            return node

        root_node = scan_node(root_dir)

        if root_node and recursive:
            def filter_node(node: Dict[str, Any]) -> bool:
                node["subdirs"] = [s for s in node["subdirs"] if filter_node(s)]
                return len(node["files"]) > 0 or len(node["subdirs"]) > 0

            filter_node(root_node)

        if root_node:
            def collect_all(node: Dict[str, Any]) -> List[Dict[str, Any]]:
                f_list = []
                f_list.extend(node["files"])
                for s in node["subdirs"]:
                    f_list.extend(collect_all(s))
                return f_list

            self.file_list = collect_all(root_node)
        else:
            self.file_list = []

        return root_node

    def scan_and_refresh(self) -> None:
        """Scan directory and rebuild UI tree."""
        self.stop_queue()
        self.file_ui_map = {}
        self.file_path_to_tag = {}
        self._tag_counter = 0

        if not self.folder or not os.path.isdir(self.folder):
            if dpg.does_item_exist(self.status_text_tag):
                dpg.set_value(self.status_text_tag, "Folder not found.")
                dpg.configure_item(self.status_text_tag, color=(255, 120, 120))
            return

        self.file_tree = self.scan_directory(self.folder, recursive=self.recursive)

        if dpg.does_item_exist(self.child_win_tree_tag):
            dpg.delete_item(self.child_win_tree_tag)

        dpg.push_container_stack(self.tree_group_tag)
        with dpg.child_window(tag=self.child_win_tree_tag, border=True):
            if self.file_tree and (len(self.file_tree["files"]) > 0 or len(self.file_tree["subdirs"]) > 0):
                self._build_tree_ui(self.file_tree, self.child_win_tree_tag)
                if dpg.does_item_exist(self.status_text_tag):
                    count = len(self.file_list)
                    dpg.set_value(
                        self.status_text_tag,
                        f"Found {count} file{'s' if count != 1 else ''} matching '{self.pattern}'.",
                    )
                    dpg.configure_item(self.status_text_tag, color=(120, 220, 140))
            else:
                dpg.add_text(f"No files matching '{self.pattern}' found.", color=(220, 140, 100))
                if dpg.does_item_exist(self.status_text_tag):
                    dpg.set_value(self.status_text_tag, "No matching files found.")
                    dpg.configure_item(self.status_text_tag, color=(220, 140, 100))
        dpg.pop_container_stack()

    def _build_tree_ui(self, node: Dict[str, Any], parent_tag: str) -> None:
        """Recursively build tree nodes and file selectables in DearPyGui."""
        if not node:
            return

        if node.get("type") == "folder":
            node_tag = f"fse_dir_{self.UUID}_{self._next_id()}"
            with dpg.tree_node(label=f"\uf07b  {node['name']}", tag=node_tag, parent=parent_tag, default_open=True):
                with dpg.popup(parent=node_tag, mousebutton=dpg.mvMouseButton_Right):
                    dpg.add_menu_item(
                        label="\uf064  Send folder files",
                        callback=self._on_send_folder_context_clicked,
                        user_data=node["path"],
                    )

                for subdir in node.get("subdirs", []):
                    self._build_tree_ui(subdir, node_tag)

                for file_info in node.get("files", []):
                    self._add_file_selectable(file_info, node_tag)
        else:
            self._add_file_selectable(node, parent_tag)

    def _add_file_selectable(self, file_info: Dict[str, Any], parent_tag: str) -> None:
        """Add a single selectable file item with right-click menu."""
        file_tag = f"fse_file_{self.UUID}_{self._next_id()}"
        self.file_ui_map[file_tag] = file_info
        self.file_path_to_tag[file_info["path"]] = file_tag

        dpg.add_selectable(
            label=f"\uf15b  {file_info['name']}",
            tag=file_tag,
            parent=parent_tag,
            callback=self._on_file_clicked,
        )

        with dpg.popup(parent=file_tag, mousebutton=dpg.mvMouseButton_Right):
            dpg.add_menu_item(
                label="\uf064  Send path",
                callback=self._on_send_single_file_context_clicked,
                user_data=file_tag,
            )

    def _on_file_clicked(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Handle selection highlight on file click."""
        if self.last_selectable and self.last_selectable != sender:
            try:
                dpg.set_value(self.last_selectable, False)
            except Exception:
                pass
        self.last_selectable = sender
        if sender and dpg.does_item_exist(sender):
            dpg.set_value(sender, True)

    def _emit_file_path(self, path: str, basename: str) -> None:
        """Emits file path on File Path output connection."""
        self.current_processing_path = path

        tag = self.file_path_to_tag.get(path)
        if tag and dpg.does_item_exist(tag):
            if self.last_selectable and self.last_selectable != tag and dpg.does_item_exist(self.last_selectable):
                try:
                    dpg.set_value(self.last_selectable, False)
                except Exception:
                    pass
            try:
                dpg.set_value(tag, True)
                self.last_selectable = tag
                dpg.bind_item_theme(tag, self.theme_processing_blue)
            except Exception:
                pass

        connections = self.connections.get("File Path", [])
        for module in connections:
            try:
                module.input_cb(data=(path, basename), data_type=IOTypes.FILE_PATH)
            except Exception as e:
                logger.error(f"[{self.label}] Emission failed for {path}: {e}")

    def _on_send_single_file_context_clicked(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Send a single file from context menu."""
        file_tag = user_data
        file_info = self.file_ui_map.get(file_tag)
        if file_info:
            self.stop_queue()
            path = file_info["path"]
            basename = file_info["basename"]
            if dpg.does_item_exist(self.status_text_tag):
                dpg.set_value(self.status_text_tag, f"Emitted: {file_info['name']} (waiting for trigger)...")
                dpg.configure_item(self.status_text_tag, color=(100, 180, 255))
            self._emit_file_path(path, basename)

    def _on_send_folder_context_clicked(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Queue and send all files in selected folder from context menu."""
        folder_path = user_data
        folder_node = self.folder_map.get(folder_path)
        if folder_node:
            def collect_all(node: Dict[str, Any]) -> List[Dict[str, Any]]:
                f_list = []
                f_list.extend(node["files"])
                for s in node["subdirs"]:
                    f_list.extend(collect_all(s))
                return f_list

            folder_files = collect_all(folder_node)
            self.start_queue(folder_files)

    def _on_send_all_clicked(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Queue and send all discovered files sequentially."""
        if not self.file_list:
            if dpg.does_item_exist(self.status_text_tag):
                dpg.set_value(self.status_text_tag, "No files to send.")
                dpg.configure_item(self.status_text_tag, color=(255, 140, 100))
            return
        self.start_queue(self.file_list)

    def _on_stop_clicked(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Stop active emission queue."""
        self.stop_queue()
        if dpg.does_item_exist(self.status_text_tag):
            dpg.set_value(self.status_text_tag, "Queue stopped by user.")
            dpg.configure_item(self.status_text_tag, color=(255, 160, 100))

    def _on_reset_status_clicked(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Reset theme and selection status of all file items in tree."""
        self.stop_queue()
        for tag in self.file_ui_map.keys():
            if dpg.does_item_exist(tag):
                try:
                    dpg.bind_item_theme(tag, 0)
                    dpg.set_value(tag, False)
                except Exception:
                    pass
        if dpg.does_item_exist(self.status_text_tag):
            dpg.set_value(self.status_text_tag, "Status reset. Ready.")
            dpg.configure_item(self.status_text_tag, color=(160, 200, 255))

    def start_queue(self, queue_items: List[Dict[str, Any]]) -> None:
        """Start sequential processing of queue items."""
        if not queue_items:
            return

        with self.process_lock:
            self.process_queue = list(queue_items)
            self.process_queue_index = 0
            self.is_queue_active = True

            if dpg.does_item_exist(self.send_all_btn_tag):
                dpg.configure_item(self.send_all_btn_tag, enabled=False)
            if dpg.does_item_exist(self.stop_btn_tag):
                dpg.configure_item(self.stop_btn_tag, enabled=True)

            first_file = self.process_queue[0]
            path = first_file["path"]
            basename = first_file["basename"]

            if dpg.does_item_exist(self.status_text_tag):
                dpg.set_value(
                    self.status_text_tag,
                    f"Processing 1/{len(self.process_queue)}: {first_file['name']} (waiting for trigger)...",
                )
                dpg.configure_item(self.status_text_tag, color=(100, 180, 255))

            self._emit_file_path(path, basename)

    def _process_next_in_queue(self) -> None:
        """Process the next file in queue after receiving completion trigger."""
        with self.process_lock:
            if not self.is_queue_active or not self.process_queue:
                return

            if self.process_queue_index < len(self.process_queue):
                prev_file = self.process_queue[self.process_queue_index]
                tag = self.file_path_to_tag.get(prev_file["path"])
                if tag and dpg.does_item_exist(tag):
                    try:
                        dpg.set_value(tag, False)
                        dpg.bind_item_theme(tag, self.theme_processed_green)
                    except Exception:
                        pass

            self.process_queue_index += 1

            if self.process_queue_index < len(self.process_queue):
                next_file = self.process_queue[self.process_queue_index]
                path = next_file["path"]
                basename = next_file["basename"]

                if dpg.does_item_exist(self.status_text_tag):
                    dpg.set_value(
                        self.status_text_tag,
                        f"Processing {self.process_queue_index + 1}/{len(self.process_queue)}: {next_file['name']} (waiting for trigger)...",
                    )
                    dpg.configure_item(self.status_text_tag, color=(100, 180, 255))

                self._emit_file_path(path, basename)
            else:
                self.is_queue_active = False
                self.current_processing_path = ""
                if dpg.does_item_exist(self.send_all_btn_tag):
                    dpg.configure_item(self.send_all_btn_tag, enabled=True)
                if dpg.does_item_exist(self.stop_btn_tag):
                    dpg.configure_item(self.stop_btn_tag, enabled=False)

                if dpg.does_item_exist(self.status_text_tag):
                    dpg.set_value(
                        self.status_text_tag,
                        f"Complete - {len(self.process_queue)} files processed successfully.",
                    )
                    dpg.configure_item(self.status_text_tag, color=(100, 255, 120))

                trigger_conns = self.connections.get("Trigger", [])
                for module in trigger_conns:
                    try:
                        module.input_cb(data=self.folder, data_type=IOTypes.TRIGGER)
                    except Exception as e:
                        logger.debug(f"[{self.label}] Completion trigger failed: {e}")

    def stop_queue(self) -> None:
        """Stop the running queue."""
        with self.process_lock:
            self.is_queue_active = False
            self.process_queue = []
            self.process_queue_index = 0
            if dpg.does_item_exist(self.send_all_btn_tag):
                dpg.configure_item(self.send_all_btn_tag, enabled=True)
            if dpg.does_item_exist(self.stop_btn_tag):
                dpg.configure_item(self.stop_btn_tag, enabled=False)

    def input_cb(self, *args: Any, **kwargs: Any) -> bool:
        """
        Handle incoming triggers, paths, or commands from upstream modules.
        """
        data = kwargs.get("data") if "data" in kwargs else (args[0] if args else None)
        data_type = kwargs.get("data_type", None)

        # Handle Trigger input (received from downstream processing module)
        if data_type == IOTypes.TRIGGER:
            if self.current_processing_path:
                tag = self.file_path_to_tag.get(self.current_processing_path)
                if tag and dpg.does_item_exist(tag):
                    try:
                        dpg.set_value(tag, False)
                        dpg.bind_item_theme(tag, self.theme_processed_green)
                    except Exception:
                        pass

            if self.is_queue_active:
                threading.Thread(target=self._process_next_in_queue, daemon=True).start()
            else:
                if dpg.does_item_exist(self.status_text_tag):
                    dpg.set_value(
                        self.status_text_tag,
                        f"Done: {os.path.basename(self.current_processing_path)}"
                        if self.current_processing_path
                        else "Trigger received.",
                    )
                    dpg.configure_item(self.status_text_tag, color=(100, 255, 120))
                self.current_processing_path = ""
            return True

        # Handle Folder Path input
        if data_type == IOTypes.FOLDER_PATH or (isinstance(data, str) and os.path.isdir(data)):
            folder = data[0] if isinstance(data, (tuple, list)) else data
            if isinstance(folder, str) and os.path.isdir(folder):
                self.folder = folder
                if dpg.does_item_exist(self.folder_input_tag):
                    dpg.set_value(self.folder_input_tag, folder)
                self.scan_and_refresh()
                return True

        # Handle File Path input (e.g. set folder to file's directory)
        if data_type == IOTypes.FILE_PATH or (isinstance(data, str) and os.path.isfile(data)):
            file_p = data[0] if isinstance(data, (tuple, list)) else data
            if isinstance(file_p, str) and os.path.exists(file_p):
                folder = os.path.dirname(file_p)
                if folder and os.path.isdir(folder):
                    self.folder = folder
                    if dpg.does_item_exist(self.folder_input_tag):
                        dpg.set_value(self.folder_input_tag, folder)
                    self.scan_and_refresh()
                    return True

        # Handle command dict
        if data_type == IOTypes.CMD_DICT and isinstance(data, dict):
            if data.get("send_all") or data.get("start"):
                self._on_send_all_clicked()
            elif data.get("stop"):
                self._on_stop_clicked()
            elif data.get("reset"):
                self._on_reset_status_clicked()
            return True

        return False

    def on_close(self) -> None:
        """Clean up resources on window close."""
        self.stop_queue()
        super().on_close()


EXPORTED_CLASS = FileScannerEmitter_win
EXPORTED_NAME = "File Scanner Emitter"
