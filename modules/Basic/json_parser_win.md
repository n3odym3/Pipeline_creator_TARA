# JSON Parser Module

A text processing utility that converts raw JSON strings into structured Python dictionary commands (`CMD_DICT`).

---

## Technical Specifications

- **Inputs**:
  - `TEXT`: JSON formatted string to be parsed.
- **Outputs**:
  - `Dict` (Type: `CMD_DICT`): The resulting parsed dictionary forwarded downstream.

---

## Usage

- Connect an incoming text stream (e.g. from an automation stream, file scanner, or network node).
- The node parses valid JSON notation, previews the formatted JSON text, and reports syntax errors with colored status indicators (`OK`, `Valid JSON but not dict`, or `JSON Error`).
