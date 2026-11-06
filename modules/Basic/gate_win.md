# Gate Module

A pass-through gate window module to dynamically route or block data streams across the pipeline directly from the UI.

---

## Technical Specifications

- **Inputs**: `In` (Type: `ANY` - dynamically inferred from connections)
- **Outputs**: `Out` (Type: `ANY` - dynamically matches input stream)

---

## Features

- **UI Control**: Toggle the pass-through state at any time via the checkbox in the window.
- **Status Indicator**: Clear visual feedback showing whether data is currently passing (OPEN) or blocked (CLOSED).
- **Dynamic Typing**: Inferred IOType automatically adjusts based on connected upstream/downstream nodes.
- **Node Editor Integration**: Fully integrated with the Node Editor's downstream highlighting, single-input constraint, and context menus.
- **Persistence**: Pass-through state and window geometry are saved and restored automatically with workspace pipelines.

---

## Usage

1. Add the **Gate** module from the `basic_ui` category or search for "Gate" in the Node Editor.
2. Connect an upstream data source (e.g., camera, generator, sensor) to the **In** terminal.
3. Connect one or more downstream consumers (e.g., image viewer, processor, plot) to the **Out** terminal.
4. Check or uncheck **Pass-through** in the Gate window to enable or block the data stream in real time.
