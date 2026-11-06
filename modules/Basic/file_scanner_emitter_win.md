# File Scanner Emitter

A batch file scanner and sequential path emitter module for DearPyGui Pipeline Creator.

---

## Technical Specifications

- **Inputs**:
  - `TRIGGER`: Completion trigger received from downstream nodes to advance to the next file in the queue.
  - `FOLDER_PATH`: Sets root scanning directory.
  - `FILE_PATH`: Sets root scanning directory to the file's parent folder.
  - `CMD_DICT`: Accepts automation commands (`send_all`/`start`, `stop`, `reset`).
- **Outputs**:
  - `File Path` (Type: `FILE_PATH`): Emits `(path, basename)` tuple for each matching file.
  - `Trigger` (Type: `TRIGGER`): Emits a completion trigger when all queued files are processed.

---

## Features
- **Directory Browsing**: Select a target folder via native OS file explorer or path input.
- **Pattern Filtering**: Filter files using wildcard patterns (`*.csv`), substrings (`_radial_detections.csv`), or exact extensions.
- **Recursive Scanning**: Toggle recursive subfolder scanning with interactive collapsible folder trees.
- **Right-Click Context Menus**:
  - Right-click on a **File**: `Send Path` to emit its path to downstream processing modules.
  - Right-click on a **Folder**: `Send Folder Files` to queue and sequentially process all files in that directory.
- **Send ALL**: Sequential processing of all matched files.
- **Trigger-Synchronized Emission**: Emits the next file **only when a completion `IOTypes.TRIGGER`** is received from downstream modules.
- **Visual State Themes**:
  - 🔵 **Blue**: Currently emitting / in processing.
  - 🟢 **Green**: Processing complete.
