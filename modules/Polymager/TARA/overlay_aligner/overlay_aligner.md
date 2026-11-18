# Overlay Aligner Module

The **Overlay Aligner** (`OverlayAlignerWin`) is a high-performance image alignment, false-colour colorization, and compositing module for DearPyGui. It superimposes two images (Layer A and Layer B) of potentially different resolutions, aspect ratios, dynamic depths (e.g. 8-bit RGB camera frames and 12-bit/16-bit FvFm fluorescence frames), and lens distortions.

It features manual geometric transformations (zoom, aspect, rotation, translation, flips, radial aberration correction), semi-manual feature-based alignment (Translation, Similarity, Affine, Homography), dynamic colormapping, transparency threshold gating, live visual footprint overlays, and JSON settings export/import.

---

## Technical Specifications

- **Accepted Inputs**:
  - `FRAME` (`IOTypes.FRAME`, `np.ndarray`): Standard 8-bit grayscale, BGR, or BGRA/RGBA images.
  - `FRAME12`, `FRAME16` (`IOTypes.FRAME12`, `IOTypes.FRAME16`, `np.ndarray`): 12-bit or 16-bit high dynamic range images (e.g. FvFm 0–1000 from Fluo Selector).
  - `FRAME_MASK_PAIR` (`IOTypes.FRAME_MASK_PAIR`, tuple of `(frame, mask)`): Base image and its associated alpha mask.
  - `FILE_PATH`, `FOLDER_PATH` (`IOTypes.FILE_PATH`, `IOTypes.FOLDER_PATH`): Image file paths to load or directory to save outputs to.
  - `TEXT` (`IOTypes.TEXT`, `str`): Group name prefix used for export file naming.
  - `CMD_DICT` (`IOTypes.CMD_DICT`, `dict`): Control commands (e.g. `{"action": "clear"}`, `{"action": "wait_a"}`, `{"action": "wait_b"}`, `{"action": "emit"}`).
- **Outputs**:
  - `RGBA` (`IOTypes.FRAME`): Composite 8-bit 4-channel BGRA image with non-premultiplied straight alpha. Pixels outside image bounds or masked out are black with `alpha = 0`.

---

## Key Features & Controls

### 1. Dual Layer Ingestion & Arming
- **WAIT A / WAIT B**: Arms the module to route the next incoming frame specifically to Layer A or Layer B (indicated by yellow button highlighting; green when populated; red when empty).
- **Clear & Swap**: Resets loaded frames or swaps A and B.
- **Load A... / Load B...**: Direct native file dialog to load images from disk (.png, .jpg, .tif, .tiff, .bmp).

### 2. Manual & Radial Geometry Alignment (Collapsible Section)
- **Geometry Dropdown**: Zoom, Aspect, Rotation, Translation, Flips, and Radial aberration parameters grouped into a collapsible header.
- **Zoom (`scale`) & Aspect (`aspect`)**: Independent scaling along axes with sub-pixel precision.
- **Rotation (`rot`)**: Smooth rotation in degrees around the footprint centre.
- **Translation (`shift_x`, `shift_y`)**: Sub-pixel horizontal and vertical shift.
- **Flips (`flip_h`, `flip_v`)**: Horizontal and vertical mirroring about the image centre.
- **Spherical / Radial Lens Correction (`k1`, `k2`)**: Corrects barrel and pincushion optical distortion using polynomial radial mapping:
  $$\vec{x}' = \vec{x}_c + (\vec{x} - \vec{x}_c)(1 + k_1 r^2 + k_2 r^4)$$
- **Nudge Controls**: Step-by-step directional buttons (`<`, `>`, `^`, `v`, `rot-`, `rot+`) with customizable step size in pixels and degrees.

### 3. Semi-Manual Point-Pair Feature Fitting (**Fit** Tab)
Because template matching and automated feature descriptors often fail between disparate modalities (e.g., standard RGB macroscopic photography vs false-colour chlorophyll fluorescence maps), a semi-manual landmark system is provided:
1. Switch to **A** view, inspect recognizable features (e.g. leaf veins, petiole nodes, plate corners), and <kbd>Ctrl</kbd> + Left Click (or Middle Click) to drop landmark points ($A_1, A_2, \dots$).
2. Switch to **B** view and click the corresponding landmarks ($B_1, B_2, \dots$) in the same sequence.
3. **Interactive Pairs Table**:
   - Displays all picked points in an interactive table (`#`, `Point A`, `Point B`, `x`).
   - Clicking any point cell highlights it in yellow and on the image plot with an illuminated halo circle and crosshair.
   - Clicking on the image while a point is selected replaces that point's coordinates directly.
   - Clicking the `x` column deletes the point pair.
4. Select the transformation model:
   - **Translation** (1+ points): Pure $XY$ shift.
   - **Similarity** (2+ points): Translation, uniform zoom, and rotation (rigid transformation).
   - **Affine** (3+ points): Translation, non-uniform scaling, rotation, and shear.
   - **Homography** (4+ points): Full projective 8-DOF planar homography.
5. Click **Fit**:
   - **Default Mode** (`Fine tune` unchecked): The fitted transformation parameters (zoom, aspect, rotation, shift) are directly written into the Layer geometry sliders.
   - **Fine Tune Mode** (`Fine tune` checked): The fitted transformation matrix is stored in the background and sliders remain at neutral/0, allowing fine-tuning adjustments on top of the fit.

### 4. Mono Processing, Colorization & Transparency Thresholds
- **Mono Channel Selection**: For RGB images, select the channel (`Gray`, `Red`, `Green`, `Blue`) used for monochrome signal processing, transparency cuts, and mask calculations. Grayscale inputs natively use their sensor data.
- **Colorization vs. Visualization**: Color images can be colorized with a false-colour palette (`colorize = True`) or displayed in their natural RGB colors (`colorize = False`) while still using the selected mono channel for mask calculations.
- **Colormaps**: `Gray`, `Inferno`, `Magma`, `Plasma`, `Viridis`, `Turbo`, `Jet`, `HSV`, `Hot`, `Green`, `Red`, `Blue`.
- **Min / Max Rescaling (`vmin`, `vmax`)**: Clamps and remaps intensity ranges, with a one-click **Auto range** button based on data extrema of the active channel.
- **Transparency Gating (`cut_low`, `cut_high`)**: Defines lower and upper thresholds below or above which pixels are rendered completely transparent (`alpha = 0`), ideal for removing non-fluorescent backgrounds or non-tissue areas.
- **Layer Opacity & Blend Modes**: Controls layer transparency and compositing math (`Normal`, `Multiply`, `Screen`, `Difference`), with informative in-UI tooltips explaining each mode.
- **Processed as Mask**: Modulates the blend weight of Layer B over Layer A using the processed mono channel (`Layer A`, `Layer B`, or `Both (A * B)`). This allows showing Layer A in its true RGB colors (e.g. green leaf tissue) while strictly restricting fluorescence (Layer B) to the leaf mask.
- **Transfer Alpha (Cross-layer Alpha Masking)**: Directly transfers the segmentation or alpha mask of one image to the other (`Apply Alpha A -> B`, `Apply Alpha B -> A`, or `Both (A * B)`).
  - *Example*: When Layer A is a SAM-extracted RGBA image (or comes from `FRAME_MASK_PAIR`), `Apply Alpha A -> B` masks diffuse fluorescence in Layer B strictly to the subject of interest.
  - *Intersection*: If Layer B has **Transparent below** (`cut_low_on`) or other cuts active, both masks superpose/multiply: background diffuse fluo outside A is transparent, and pixels below threshold in B also become transparent.

### 5. Canvas & Overlays
- **View Modes**:
  - `Overlay`: Both layers transformed, aligned, colorized, and composited onto the world canvas.
  - `A raw`: Layer A shown in its native source space completely unprocessed (raw RGB colors or linear grayscale).
  - `A processed`: Layer A shown in its native source space with all user processing applied (channel selection, vmin/vmax, colormap, invert, threshold cuts).
  - `B raw`: Layer B shown in its native source space completely unprocessed (raw sensor data).
  - `B processed`: Layer B shown in its native source space with all user processing applied (false-color palette, vmin/vmax, transparency cuts).
- **Footprint Contours**: Renders color-coded boundary polylines (cyan for A, magenta for B) showing exact rotation, scaling, shifting, and radial bowing of each frame.
- **Grid Overlay**: Configurable alignment grid with adjustable subdivisions.
- **Canvas Modes**:
  - `A`: Output dimensions match Layer A native resolution.
  - `B`: Output dimensions match Layer B native resolution.
  - `Union`: Minimal bounding box enclosing both transformed images in world space.
  - `Custom`: User-defined width and height.
- **Backgrounds**: `Checker`, `Dark`, `Black`, `White`, `Custom` (with customizable color picker).

### 6. Settings Export & Automation
- **JSON Serialization**: Full alignment parameters (geometry, radial coefficients, fit matrices, picked landmarks, color thresholds, canvas settings) can be saved to or loaded from `.json` files.
- **Workspace Persistence**: All alignment parameters persist across Pipeline Creator project save and load cycles.
- **Auto-emit**: Simple checkbox that automatically emits the currently visible frame downstream whenever updated.
- **Background Checkbox**: When checked, composites the output over the active view background (including custom color) instead of outputting transparent alpha.

---

## Pipeline Integration

```mermaid
graph LR
    Cam[Color Camera / File] -->|FRAME (Slot A)| Aligner[Overlay Aligner]
    Fluo[Fluo Selector FvFm] -->|FRAME12 (Slot B)| Aligner
    Aligner -->|RGBA| Viewer[Image Viewer / RGBA Display]
    Aligner -->|RGBA| Downstream[Export / Analysis Node]
```
