# Simple Button Module

A manual trigger source to kick off automation cycles, calibrations, or camera captures.

---

## Technical Specifications

- **Inputs**:
  - `TRIGGER`: Programmatic trigger forwarded downstream.
- **Outputs**:
  - `Trig` (Type: `TRIGGER`): Emits a standard execution trigger signal.

---

## Usage

- Click the **Trigger** button in the UI window to emit a trigger signal to connected nodes.
- Upstream programmatic triggers are forwarded through the module.
