<div align="center">

# Polymager TARA

**Field chlorophyll fluorescence and coral bleaching imaging application for CBASS experiments during the *Tara* Coral expedition.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![GUI](https://img.shields.io/badge/GUI-DearPyGui-ff69b4.svg)](https://github.com/hoffstadt/DearPyGui)
[![Upstream](https://img.shields.io/badge/built%20on-Pipeline%20Creator-blueviolet.svg)](https://github.com/n3odym3/Pipeline_creator)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

---

## Overview

This repository is the **Tara Coral** deployment of [**Pipeline Creator**](https://github.com/n3odym3/Pipeline_creator), pre-configured as **Polymager TARA** to run standardized **CBASS** (*Coral Bleaching Automated Stress System*) thermal stress assays in the field and aboard the *Tara* schooner.

Unlike the upstream core framework, this repository ships ready-to-run with the full CBASS pipeline (`pipelines/Polymager_TARA.json`), startup automation (`scripts/polymager_TARA_script.json`), hardware drivers, fluorescence protocols, AI segmentation, and step-by-step interactive operator tutorials (`TARA Interface`).

<p align="center">
  <img src="docs/media/Polymager.png" alt="Polymager TARA CBASS Interface" width="95%">
</p>

---

## CBASS Workflows

The **TARA Interface** guides operators through three dedicated pipelines:

### 1. Data Capture
- **Structured CBASS Storage:** Creates and manages master/date folder trees (`FvFm` and `SuperSeq` subfolders) across experimental timepoints (`T0`, `T1`, `T2`, etc.) and treatment conditions.
- **Dual-Camera & Illumination Control:**
  - **IDS Camera** (`pyueye`): 12-bit monochrome chlorophyll fluorescence acquisition with dark-reference and flat-field correction.
  - **Allied Vision Alvium Camera** (`vmbpy`): High-resolution RGB colorimetry acquisition with white balance and flat-field correction.
  - **Fluo Controller** (serial microcontroller): Synchronized control of Actinic, Detection, Far-Red, and dual White LEDs with calibrated irradiance ($\mu\text{mol photons}\cdot\text{m}^{-2}\cdot\text{s}^{-1}$).
- **Automated Sequences:** Guides exposure calibration (`Auto Config Expos`), test detection checks, automated fluorescence protocols (`FvFm.txt`, `rETR.txt`, `SuperSeq.txt`), and RGB sample capture (`Take Picture`).

### 2. Data Processing — Fluorescence (`Fluo`)
- Computes pixel-wise $F_v/F_m = (F_m - F_0) / F_m$ maps and extracts ROI contours (`Fluo Selector`), or processes fluorescence kinetics across ROIs (`Image Container`, `ROI Manager`, `Nodal Data Processor`).
- Attaches standardized **OD50** metadata (`Project`, `Date`, `Country`, `Site`, `Species`, `Genotype`, `Condition`, `Temperature`, `Timepoint`, etc.) and exports aggregated CSV tables via `TARA Agregator`.

### 3. Data Processing — Visual Bleaching Assessment (`VBA`)
- **AI Coral Segmentation:** Interactive segmentation using Meta's **Segment Anything Model** (`SAM Interactive`, `vit_b` with CUDA/CPU support) via positive/negative point prompts (`CTRL + Left/Right Click`).
- **Color Quantification:** Classifies pixel colors inside the segmented coral mask against a calibrated reference palette (`Color Mapper`), exports cropped `_RAW.png` and `_processed.png` images, and aggregates color distributions with OD50 metadata via `TARA Color Agregator`.

---

## Field-Ready Operator Setup

- **Automatic Startup:** `config/config.json` launches directly into **Polymager TARA** in `user` mode (locked-down operator UI) and runs `scripts/polymager_TARA_script.json` to load `pipelines/Polymager_TARA.json` and initialize the cameras and Fluo Controller.
- **Built-in Interactive Tutorials:** Step-by-step guided walkthroughs (`modules/Polymager/TARA/TARA_helper/step*.json`) highlight UI elements and enforce checks so rotating science crews can run CBASS protocols consistently.
- **Advanced / Dev Modes:** Privilege elevation (`Advanced` / `Dev`) remains available from the login menu to edit node wiring, calibrate LEDs, or customize sequences.

---

## Quick Start

### 1. Clone & Install Python Dependencies

```bash
git clone https://github.com/n3odym3/Pipeline_creator_TARA.git
cd Pipeline_creator_TARA

# Core GUI dependencies
pip install -r requirements.txt
```

For full hardware and processing support, ensure the module dependencies and vendor camera SDKs are installed:
- **Drivers / SDKs:** IDS **uEye** (`pyueye`) and Allied Vision **Vimba X** (`vmbpy`)
- **Hardware & Processing:** `pyserial`, `opencv-python`, `torch`, `segment-anything`


> The SAM Interactive module expects the `vit_b` checkpoint (`modules/computer_vision/sam_interactive/sam_vit_b_01ec64.pth`).

### 2. Launch the Application

```bash
python main.py
```

### 3. Windows Standalone Build (Optional)

`make_executable.py` (Nuitka) and `setup.iss` (Inno Setup) are included to compile `Polymager.exe` and package a Windows installer bundling the IDS uEye and Allied Vision Vimba X drivers.

---

## Upstream Framework & License

- **Upstream Project:** [Pipeline Creator](https://github.com/n3odym3/Pipeline_creator) — [Framework Documentation](https://n3odym3.github.io/Pipeline_creator/)
- **License:** [MIT License](LICENSE)
