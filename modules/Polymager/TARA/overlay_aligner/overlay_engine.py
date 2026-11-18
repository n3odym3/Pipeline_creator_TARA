"""
Overlay Aligner – processing engine.

Pure NumPy / OpenCV code (no DearPyGui dependency) used by
``overlay_aligner_win.OverlayAlignerWin``. Keeping it GUI-free makes it unit-testable.

Coordinate conventions
----------------------
* Image space: x to the right, y downwards, **pixel centres at integer coordinates**
  (pixel ``i`` covers ``[i - 0.5, i + 0.5]``) – the OpenCV convention.
* "World" space: the common space in which both layers are placed. Layer A is placed
  at its native resolution (identity) unless the user moves it.
* Canvas: a rectangular window onto world space (``origin`` + size) at a given scale.

Per-layer transform chain (source pixel -> world)::

    world = P · H_fit · F · U(src)

* ``U``     : radial ("spherical") distortion correction (k1, k2) around the image centre.
* ``F``     : optional horizontal / vertical flip around the image centre.
* ``H_fit`` : 3x3 matrix computed from user-picked point pairs (identity by default).
* ``P``     : manual fine adjustment (zoom, aspect, rotation, shift) applied about the
              centre of the layer footprint.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants / option lists
# ─────────────────────────────────────────────────────────────────────────────
MAX_CANVAS = 12000
MAX_EXPORT_PIXELS = 64_000_000

PALETTES = ["Gray", "Inferno", "Magma", "Plasma", "Viridis", "Turbo", "Jet", "HSV", "Hot", "Green", "Red", "Blue"]
BLEND_MODES = ["Normal", "Multiply", "Screen", "Difference"]
CANVAS_MODES = ["A", "B", "Union", "Custom"]
VIEW_MODES = ["Overlay", "A raw", "A processed", "B raw", "B processed"]
BACKGROUNDS = ["Checker", "Dark", "Black", "White", "Custom"]
FIT_MODELS = ["Translation", "Similarity", "Affine", "Homography"]
FIT_DIRECTIONS = ["B -> A", "A -> B"]
MIN_POINTS = {"Translation": 1, "Similarity": 2, "Affine": 3, "Homography": 4}
MONO_CHANNELS = ["Gray", "Red", "Green", "Blue"]
MASK_SOURCES = ["Layer A", "Layer B", "Both (A * B)"]
ALPHA_TRANSFERS = ["None", "Apply Alpha A -> B", "Apply Alpha B -> A", "Both (A * B)"]

GEOMETRY_KEYS = ("scale", "aspect", "rot", "shift_x", "shift_y", "flip_h", "flip_v", "k1", "k2")
FINE_GEOMETRY_KEYS = ("scale", "aspect", "rot", "shift_x", "shift_y")

SETTINGS_VERSION = 1


def default_layer_params(name: str) -> dict[str, Any]:
    """Default per-layer parameters. Layer B defaults to a false-colour fluorescence look."""
    p: dict[str, Any] = {
        "visible": True,
        "opacity": 1.0,
        "mono_channel": "Gray",
        "colorize": False,
        "palette": "Gray",
        "invert": False,
        "vmin": 0.0,
        "vmax": 255.0,
        "cut_low_on": False,
        "cut_low": 0.0,
        "cut_high_on": False,
        "cut_high": 1000.0,
        "scale": 1.0,
        "aspect": 1.0,
        "rot": 0.0,
        "shift_x": 0.0,
        "shift_y": 0.0,
        "flip_h": False,
        "flip_v": False,
        "k1": 0.0,
        "k2": 0.0,
    }
    if name == "B":
        p.update(colorize=True, palette="Inferno", vmin=0.0, vmax=1000.0, cut_low_on=True, cut_low=1.0)
    return p


def default_global_params() -> dict[str, Any]:
    """Default global (non per-layer) parameters."""
    return {
        "blend": "Normal",
        "transfer_alpha": "None",
        "processed_mask": False,
        "mask_source": "Layer A",
        "show_scale": False,
        "scale_pos": "Right",
        "scale_source": "Layer B",
        "scale_mult": 0.001,
        "scale_custom_min": 0.0,
        "scale_custom_max": 1.0,
        "view_mode": "Overlay",
        "background": "Checker",
        "bg_custom": [0.0, 0.0, 0.0],
        "export_background": False,
        "show_frames": True,
        "show_grid": False,
        "grid_div": 8,
        "show_points": True,
        "canvas_mode": "A",
        "canvas_w": 1920,
        "canvas_h": 1080,
        "preview_max": 1200,
        "export_scale": 1.0,
        "auto_emit": True,
        "live_emit": False,
        "auto_save": False,
        "fit_model": "Similarity",
        "fit_dir": "B -> A",
        "fine_tune": False,
    }


_ALLOWED_STRINGS = {
    "palette": PALETTES,
    "blend": BLEND_MODES,
    "view_mode": VIEW_MODES,
    "background": BACKGROUNDS,
    "canvas_mode": CANVAS_MODES,
    "fit_model": FIT_MODELS,
    "fit_dir": FIT_DIRECTIONS,
    "mono_channel": MONO_CHANNELS,
    "mask_source": MASK_SOURCES,
    "transfer_alpha": ALPHA_TRANSFERS,
    "scale_pos": ["Right", "Left"],
    "scale_source": ["Layer B", "Layer A", "Custom"],
}


def _coerce(key: str, default: Any, value: Any) -> Any:
    """Coerce ``value`` to the type of ``default`` (invalid values fall back to default)."""
    try:
        if isinstance(default, bool):
            return bool(value)
        if isinstance(default, int):
            return int(value)
        if isinstance(default, float):
            v = float(value)
            return v if math.isfinite(v) else default
        if isinstance(default, str):
            s = str(value)
            allowed = _ALLOWED_STRINGS.get(key)
            return s if (allowed is None or s in allowed) else default
        if isinstance(default, list):
            return list(value)
    except (TypeError, ValueError):
        pass
    return default


# ─────────────────────────────────────────────────────────────────────────────
# Source image container
# ─────────────────────────────────────────────────────────────────────────────
_version_counter = itertools.count(1)


def _to_u8(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint8:
        return a
    f = a.astype(np.float32)
    mx = float(f.max()) if f.size else 1.0
    if mx <= 1.0:
        f *= 255.0
    elif mx > 255.0:
        f *= 255.0 / mx
    return np.clip(f, 0, 255).astype(np.uint8)


class LayerImage:
    """
    Pre-processed source image ready to be warped.

    * ``kind == "gray"``  : ``stack`` is float32 HxWx2 = [value*coverage, coverage].
      Warping both channels together gives an exact "normalised" interpolation at borders.
    * ``kind == "color"`` : ``stack`` is uint8 HxWx4 = BGRA.
    """

    def __init__(self, img: np.ndarray, alpha: np.ndarray | None = None) -> None:
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        if arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[2] not in (3, 4)):
            raise ValueError(f"Unsupported image shape {arr.shape}")

        self.version: int = next(_version_counter)
        self.h, self.w = int(arr.shape[0]), int(arr.shape[1])
        if self.h < 2 or self.w < 2:
            raise ValueError("Image too small")

        cov_src: np.ndarray | None = None
        if alpha is not None:
            a = np.asarray(alpha)
            if a.ndim == 3:
                a = a[:, :, 0]
            if a.dtype == bool:
                af = a.astype(np.float32)
            else:
                af = a.astype(np.float32)
                if af.size and af.max() > 1.0:
                    af = af / 255.0
            if af.shape != (self.h, self.w):
                af = cv2.resize(af, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
            cov_src = np.clip(af, 0.0, 1.0)

        if arr.ndim == 2:
            self.kind = "gray"
            sc = np.nan_to_num(arr.astype(np.float32))
            cov = cov_src if cov_src is not None else np.ones((self.h, self.w), np.float32)
            self.dmin = float(sc.min())
            self.dmax = float(sc.max())
            self.stack = np.ascontiguousarray(np.dstack([sc * cov, cov]).astype(np.float32))
            self.desc = f"gray {arr.dtype}"
        else:
            self.kind = "color"
            bgr = _to_u8(arr[:, :, :3])
            if arr.shape[2] == 4:
                a8 = _to_u8(arr[:, :, 3])
            else:
                a8 = np.full((self.h, self.w), 255, np.uint8)
            if cov_src is not None:
                a8 = np.clip(a8.astype(np.float32) * cov_src, 0, 255).astype(np.uint8)
            self.dmin, self.dmax = 0.0, 255.0
            self.stack = np.ascontiguousarray(np.dstack([bgr, a8]))
            self.desc = f"color {arr.dtype}"

    @property
    def size(self) -> tuple[int, int]:
        return self.w, self.h


class Layer:
    """Mutable layer state: image + parameters + fit matrix + picked points."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.image: LayerImage | None = None
        self.params: dict[str, Any] = default_layer_params(name)
        self.fit_H: np.ndarray = np.eye(3)
        self.points: list[list[float]] = []
        self.range_user_set: bool = False

    def reset_fine_geometry(self) -> None:
        d = default_layer_params(self.name)
        for k in FINE_GEOMETRY_KEYS:
            self.params[k] = d[k]

    def reset_geometry(self) -> None:
        d = default_layer_params(self.name)
        for k in GEOMETRY_KEYS:
            self.params[k] = d[k]
        self.fit_H = np.eye(3)

    def has_fit(self) -> bool:
        return not np.allclose(self.fit_H, np.eye(3), atol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────
def translate(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 matrix to (N,2) points (with perspective division)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    w = p[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return p[:, :2] / w


def flip_matrix(w: int, h: int, flip_h: bool, flip_v: bool) -> np.ndarray:
    sx = -1.0 if flip_h else 1.0
    sy = -1.0 if flip_v else 1.0
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    return np.array([[sx, 0.0, cx - sx * cx], [0.0, sy, cy - sy * cy], [0.0, 0.0, 1.0]])


def _radial_norm(w: int, h: int) -> float:
    return max(w, h) / 2.0


def distort_points(u: np.ndarray, w: int, h: int, k1: float, k2: float) -> np.ndarray:
    """Undistorted -> distorted (source) coordinates."""
    u = np.asarray(u, dtype=np.float64).reshape(-1, 2)
    if k1 == 0.0 and k2 == 0.0:
        return u.copy()
    c = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    n = _radial_norm(w, h)
    d = (u - c) / n
    r2 = np.sum(d * d, axis=1, keepdims=True)
    f = 1.0 + k1 * r2 + k2 * r2 * r2
    return c + d * f * n


def undistort_points(d: np.ndarray, w: int, h: int, k1: float, k2: float, iters: int = 30) -> np.ndarray:
    """Distorted (source) -> undistorted coordinates (fixed-point inversion)."""
    d = np.asarray(d, dtype=np.float64).reshape(-1, 2)
    if k1 == 0.0 and k2 == 0.0:
        return d.copy()
    c = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    n = _radial_norm(w, h)
    dn = (d - c) / n
    u = dn.copy()
    for _ in range(iters):
        r2 = np.sum(u * u, axis=1, keepdims=True)
        f = 1.0 + k1 * r2 + k2 * r2 * r2
        f = np.where(np.abs(f) < 0.05, 0.05, f)
        u = dn / f
    return c + u * n


def pre_matrix(p: dict[str, Any], w: int, h: int) -> np.ndarray:
    """Matrix F (flip) applied after radial correction, before the fit."""
    return flip_matrix(w, h, bool(p["flip_h"]), bool(p["flip_v"]))


def user_matrix(p: dict[str, Any], fit_H: np.ndarray, w: int, h: int) -> np.ndarray:
    """Matrix P: manual zoom / aspect / rotation / shift about the footprint centre."""
    pivot = apply_h(fit_H, np.array([[(w - 1) / 2.0, (h - 1) / 2.0]]))[0]
    th = math.radians(float(p["rot"]))
    c, s = math.cos(th), math.sin(th)
    rot = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])  # visually CCW (image y is down)
    zoom = float(p["scale"])
    scl = np.diag([zoom * float(p["aspect"]), zoom, 1.0])
    t_in = translate(-pivot[0], -pivot[1])
    t_out = translate(pivot[0] + float(p["shift_x"]), pivot[1] + float(p["shift_y"]))
    return t_out @ rot @ scl @ t_in


def decompose_affine_transform(H: np.ndarray, w: int, h: int) -> dict[str, float]:
    """
    Decompose a 3x3 matrix H (mapping pre-fit coordinates to world coordinates)
    into standard geometry parameters: scale, aspect, rot (deg), shift_x, shift_y
    about the image centre ((w-1)/2, (h-1)/2).
    """
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    c_pt = np.array([[cx, cy]])
    c_world = apply_h(H, c_pt)[0]
    shift_x = float(c_world[0] - cx)
    shift_y = float(c_world[1] - cy)

    # Local differential mapping of vectors (1, 0) and (0, 1) from centre
    px = apply_h(H, np.array([[cx + 1.0, cy]]))[0] - c_world
    py = apply_h(H, np.array([[cx, cy + 1.0]]))[0] - c_world

    sx = float(np.hypot(px[0], px[1]))
    sy = float(np.hypot(py[0], py[1]))
    scale = sy
    aspect = (sx / sy) if sy > 1e-9 else 1.0

    # In user_matrix: rot = [[cos, sin], [-sin, cos]]
    # Py = sy * [sin(th), cos(th)]
    th = math.atan2(py[0], py[1])
    rot = float(math.degrees(th))

    return {
        "scale": scale,
        "aspect": aspect,
        "rot": rot,
        "shift_x": shift_x,
        "shift_y": shift_y,
    }


def world_matrix(p: dict[str, Any], fit_H: np.ndarray, w: int, h: int) -> np.ndarray:
    """Full matrix mapping *undistorted* source pixels to world coordinates."""
    return user_matrix(p, fit_H, w, h) @ fit_H @ pre_matrix(p, w, h)


def src_to_world(pts: np.ndarray, p: dict[str, Any], fit_H: np.ndarray, w: int, h: int) -> np.ndarray:
    """Raw (distorted) source pixel coordinates -> world coordinates."""
    u = undistort_points(pts, w, h, float(p["k1"]), float(p["k2"]))
    return apply_h(world_matrix(p, fit_H, w, h), u)


def src_to_pre(pts: np.ndarray, p: dict[str, Any], w: int, h: int) -> np.ndarray:
    """Raw source pixel coordinates -> 'pre-fit' space (radial correction + flip)."""
    u = undistort_points(pts, w, h, float(p["k1"]), float(p["k2"]))
    return apply_h(pre_matrix(p, w, h), u)


def footprint_world(p: dict[str, Any], fit_H: np.ndarray, w: int, h: int, n: int = 24) -> np.ndarray:
    """
    Outline of the layer in world coordinates, as a (4n, 2) polyline starting at the
    source top-left corner and going TL -> TR -> BR -> BL (n samples per edge so that
    radial distortion curves are visible).
    """
    x0, y0, x1, y1 = -0.5, -0.5, w - 0.5, h - 0.5
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    top = np.stack([x0 + (x1 - x0) * t, np.full_like(t, y0)], axis=1)
    right = np.stack([np.full_like(t, x1), y0 + (y1 - y0) * t], axis=1)
    bottom = np.stack([x1 - (x1 - x0) * t, np.full_like(t, y1)], axis=1)
    left = np.stack([np.full_like(t, x0), y1 - (y1 - y0) * t], axis=1)
    edge = np.vstack([top, right, bottom, left])
    return src_to_world(edge, p, fit_H, w, h)


# ─────────────────────────────────────────────────────────────────────────────
# Canvas
# ─────────────────────────────────────────────────────────────────────────────
def compute_canvas(snaps: dict[str, dict[str, Any]], g: dict[str, Any]) -> tuple[int, int, int, int]:
    """Return ``(origin_x, origin_y, width, height)`` of the output canvas in world pixels."""
    a, b = snaps["A"], snaps["B"]
    mode = g["canvas_mode"]
    ox = oy = 0
    if mode == "Custom":
        w, h = int(g["canvas_w"]), int(g["canvas_h"])
    elif mode == "Union" and (a["image"] is not None or b["image"] is not None):
        pts = []
        for sn in (a, b):
            li = sn["image"]
            if li is not None:
                pts.append(footprint_world(sn["params"], sn["fit_H"], li.w, li.h, n=8))
        allp = np.vstack(pts)
        allp = allp[np.all(np.isfinite(allp), axis=1)]
        if len(allp) == 0:
            w, h = 640, 480
        else:
            ox = int(math.floor(allp[:, 0].min() + 0.5))
            oy = int(math.floor(allp[:, 1].min() + 0.5))
            w = int(math.ceil(allp[:, 0].max() + 0.5)) - ox
            h = int(math.ceil(allp[:, 1].max() + 0.5)) - oy
    else:
        ref = None
        if mode == "B" and b["image"] is not None:
            ref = b["image"]
        elif a["image"] is not None:
            ref = a["image"]
        elif b["image"] is not None:
            ref = b["image"]
        w, h = (ref.w, ref.h) if ref is not None else (640, 480)
    w = int(min(max(w, 1), MAX_CANVAS))
    h = int(min(max(h, 1), MAX_CANVAS))
    return ox, oy, w, h


def canvas_matrix(origin: tuple[int, int], s: float) -> np.ndarray:
    """world -> canvas pixel (at scale ``s``), pixel-centre aware."""
    return np.array(
        [
            [s, 0.0, s * (0.5 - origin[0]) - 0.5],
            [0.0, s, s * (0.5 - origin[1]) - 0.5],
            [0.0, 0.0, 1.0],
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Warping
# ─────────────────────────────────────────────────────────────────────────────
def warp_layer(
    li: LayerImage,
    p: dict[str, Any],
    fit_H: np.ndarray,
    origin: tuple[int, int],
    out_w: int,
    out_h: int,
    s: float,
) -> np.ndarray:
    """Warp the layer's pre-processed stack onto the canvas (``out_h`` x ``out_w``)."""
    def _zero() -> np.ndarray:
        return np.zeros((out_h, out_w, li.stack.shape[2]), dtype=li.stack.dtype)

    A3 = canvas_matrix(origin, s) @ world_matrix(p, fit_H, li.w, li.h)
    try:
        inv = np.linalg.inv(A3)
    except np.linalg.LinAlgError:
        return _zero()
    if not np.all(np.isfinite(inv)):
        return _zero()
    inv = inv / (inv[2, 2] if abs(inv[2, 2]) > 1e-12 else 1.0)

    k1, k2 = float(p["k1"]), float(p["k2"])
    flags = cv2.INTER_LINEAR
    if k1 == 0.0 and k2 == 0.0:
        if abs(inv[2, 0]) < 1e-12 and abs(inv[2, 1]) < 1e-12:
            res = cv2.warpAffine(
                li.stack, inv[:2].astype(np.float64), (out_w, out_h),
                flags=flags | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
        else:
            res = cv2.warpPerspective(
                li.stack, inv.astype(np.float64), (out_w, out_h),
                flags=flags | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
    else:
        gx, gy = np.meshgrid(np.arange(out_w, dtype=np.float32), np.arange(out_h, dtype=np.float32))
        den = inv[2, 0] * gx + inv[2, 1] * gy + inv[2, 2]
        den = np.where(np.abs(den) < 1e-9, 1e-9, den)
        ux = (inv[0, 0] * gx + inv[0, 1] * gy + inv[0, 2]) / den
        uy = (inv[1, 0] * gx + inv[1, 1] * gy + inv[1, 2]) / den
        cx, cy = (li.w - 1) / 2.0, (li.h - 1) / 2.0
        n = _radial_norm(li.w, li.h)
        dx = (ux - cx) / n
        dy = (uy - cy) / n
        r2 = dx * dx + dy * dy
        f = 1.0 + k1 * r2 + k2 * r2 * r2
        map_x = (cx + dx * f * n).astype(np.float32)
        map_y = (cy + dy * f * n).astype(np.float32)
        res = cv2.remap(li.stack, map_x, map_y, flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if res.ndim == 2:
        res = res[:, :, None]
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Colorization / shading
# ─────────────────────────────────────────────────────────────────────────────
_LUT_CACHE: dict[str, np.ndarray] = {}


def palette_lut(name: str) -> np.ndarray:
    """256x3 uint8 BGR lookup table for a palette name."""
    if name in _LUT_CACHE:
        return _LUT_CACHE[name]
    g = np.arange(256, dtype=np.uint8)
    if name == "Gray":
        lut = np.stack([g, g, g], axis=1)
    elif name in ("Green", "Red", "Blue"):
        lut = np.zeros((256, 3), np.uint8)
        lut[:, {"Blue": 0, "Green": 1, "Red": 2}[name]] = g
    else:
        cmap = getattr(cv2, f"COLORMAP_{name.upper()}", cv2.COLORMAP_JET)
        lut = cv2.applyColorMap(g.reshape(256, 1), cmap).reshape(256, 3)
    lut = np.ascontiguousarray(lut)
    _LUT_CACHE[name] = lut
    return lut


def shade_layer(
    warped: np.ndarray, kind: str, p: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Turn a warped stack into ``(bgr uint8 HxWx3, alpha float32 HxW in 0..1, mono_mask float32 HxW in 0..1, layer_mask float32 HxW in 0..1)``.

    Applies colorization (min/max/palette/invert), transparency thresholds and opacity.
    ``mono_mask`` represents the processed mono signal (channel selection, vmin/vmax normalization,
    invert, and cut_low/cut_high thresholds) regardless of whether colorization is enabled for display.
    ``layer_mask`` represents the intrinsic spatial mask (coverage * threshold cuts, without opacity).
    """
    if kind == "gray":
        cov = warped[:, :, 1]
        safe = np.where(cov > 1e-4, cov, 1.0)
        v = np.where(cov > 1e-4, warped[:, :, 0] / safe, 0.0).astype(np.float32)
        bgr = None
    else:
        bgr = warped[:, :, :3]
        cov = warped[:, :, 3].astype(np.float32) * (1.0 / 255.0)
        ch = str(p.get("mono_channel", "Gray"))
        if ch in ("Red", "R"):
            v = bgr[:, :, 2].astype(np.float32)
        elif ch in ("Green", "G"):
            v = bgr[:, :, 1].astype(np.float32)
        elif ch in ("Blue", "B"):
            v = bgr[:, :, 0].astype(np.float32)
        else:
            v = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2GRAY).astype(np.float32)

    lo, hi = float(p["vmin"]), float(p["vmax"])
    span = hi - lo
    if abs(span) < 1e-9:
        span = 1e-9
    t = np.clip((v - lo) / span, 0.0, 1.0)
    if p["invert"]:
        t = 1.0 - t

    if kind == "gray" or p["colorize"]:
        idx = (t * 255.0 + 0.5).astype(np.uint8)
        bgr = palette_lut(str(p["palette"]))[idx]
    else:
        bgr = np.ascontiguousarray(bgr)
        if p["invert"]:
            bgr = 255 - bgr

    cov_alpha = cov.astype(np.float32)
    mask_cut = np.ones_like(cov_alpha, dtype=np.float32)
    if p["cut_low_on"]:
        mask_cut = mask_cut * (v >= float(p["cut_low"]))
    if p["cut_high_on"]:
        mask_cut = mask_cut * (v <= float(p["cut_high"]))
    layer_mask = np.ascontiguousarray(cov_alpha * mask_cut, dtype=np.float32)
    alpha = layer_mask * float(p["opacity"])

    mono_mask = np.ascontiguousarray(t * mask_cut * cov_alpha, dtype=np.float32)
    return bgr, np.ascontiguousarray(alpha, dtype=np.float32), mono_mask, layer_mask


def composite(
    layers: list[tuple[np.ndarray, np.ndarray]], blend: str, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Alpha-composite layers bottom -> top. Returns straight (non-premultiplied)
    ``(bgr float32 0..1, alpha float32 0..1)``.
    """
    out_c: np.ndarray | None = None
    out_a: np.ndarray | None = None
    for bgr, a in layers:
        cs = bgr.astype(np.float32) * (1.0 / 255.0)
        a = a.astype(np.float32)
        if out_c is None or out_a is None:
            out_c, out_a = cs, a
            continue
        ab = out_a[..., None]
        as_ = a[..., None]
        if blend == "Multiply":
            bf = out_c * cs
        elif blend == "Screen":
            bf = 1.0 - (1.0 - out_c) * (1.0 - cs)
        elif blend == "Difference":
            bf = np.abs(out_c - cs)
        else:
            bf = cs
        if blend != "Normal":
            bf = (1.0 - ab) * cs + ab * bf
        na = as_ + ab * (1.0 - as_)
        out_c = (as_ * bf + ab * (1.0 - as_) * out_c) / np.maximum(na, 1e-6)
        out_a = na[..., 0]
    if out_c is None or out_a is None:
        h, w = shape
        return np.zeros((h, w, 3), np.float32), np.zeros((h, w), np.float32)
    return out_c, out_a


_BG_CACHE: dict[tuple[int, int, str, tuple], np.ndarray] = {}


def _background(h: int, w: int, mode: str, custom_rgb: Any = None) -> np.ndarray:
    c_key = tuple(float(x) for x in custom_rgb) if (custom_rgb is not None and isinstance(custom_rgb, (list, tuple))) else ()
    key = (h, w, mode, c_key)
    cached = _BG_CACHE.get(key)
    if cached is not None and cached.shape[:2] == (h, w):
        return cached

    if mode == "Checker":
        sq = 8
        pat = (((np.arange(h) // sq)[:, None] + (np.arange(w) // sq)[None, :]) & 1).astype(np.float32)
        base = 0.20 + 0.08 * pat
        res = np.repeat(base[:, :, None], 3, axis=2)
    elif mode == "Custom":
        c = custom_rgb if custom_rgb is not None else [0.0, 0.0, 0.0]
        try:
            r, g, b = float(c[0]), float(c[1]), float(c[2])
            if max(r, g, b) > 1.0:
                r, g, b = r / 255.0, g / 255.0, b / 255.0
        except Exception:
            r, g, b = 0.0, 0.0, 0.0
        res = np.full((h, w, 3), [b, g, r], np.float32)
    else:
        val = {"Dark": 0.11, "Black": 0.0, "White": 1.0}.get(mode, 0.11)
        res = np.full((h, w, 3), val, np.float32)

    if len(_BG_CACHE) > 10:
        _BG_CACHE.clear()
    _BG_CACHE[key] = res
    return res


def to_display_rgba(bgr_f: np.ndarray, a_f: np.ndarray, background: str, custom_rgb: Any = None) -> np.ndarray:
    """Composite over a background and return a flat float32 RGBA array for a raw texture."""
    h, w = a_f.shape[:2]
    bg = _background(h, w, background, custom_rgb)
    a = a_f[..., None]
    out = bgr_f * a + bg * (1.0 - a)
    rgba = cv2.cvtColor(out, cv2.COLOR_BGR2RGBA)
    return np.ascontiguousarray(rgba).ravel()


def to_bgra(bgr_f: np.ndarray, a_f: np.ndarray, background: str | None = None, custom_rgb: Any = None) -> np.ndarray:
    """Convert to a BGRA uint8 image. If background is specified, composite over it and set alpha to 255."""
    h, w = a_f.shape[:2]
    out = np.empty((h, w, 4), np.uint8)
    if background is not None:
        bg = _background(h, w, background, custom_rgb)
        a = a_f[..., None]
        comp = bgr_f * a + bg * (1.0 - a)
        out[..., :3] = np.clip(comp * 255.0 + 0.5, 0, 255).astype(np.uint8)
        out[..., 3] = 255
    else:
        out[..., :3] = np.clip(bgr_f * 255.0 + 0.5, 0, 255).astype(np.uint8)
        out[..., 3] = np.clip(a_f * 255.0 + 0.5, 0, 255).astype(np.uint8)
        out[a_f <= 0.0, :3] = 0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Rendering entry points
# ─────────────────────────────────────────────────────────────────────────────
class RenderResult:
    """Result of an overlay render."""

    def __init__(self, bgr: np.ndarray, alpha: np.ndarray, width: int, height: int, ox: int, oy: int, s: float):
        self.bgr = bgr
        self.alpha = alpha
        self.canvas_w = width
        self.canvas_h = height
        self.origin = (ox, oy)
        self.scale = s


def draw_intensity_scale(
    bgr: np.ndarray, a: np.ndarray, g: dict[str, Any], snaps: dict[str, dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw a vertical intensity scale bar with min/max/mid numeric annotations onto the composite.
    """
    h, w = bgr.shape[:2]
    if h < 50 or w < 50:
        return bgr, a

    src_choice = str(g.get("scale_source", "Layer B"))
    if src_choice == "Layer A" and "A" in snaps:
        layer_params = snaps["A"]["params"]
        palette_name = str(layer_params.get("palette", "Gray"))
        vmin = float(layer_params.get("vmin", 0.0))
        vmax = float(layer_params.get("vmax", 255.0))
        invert = bool(layer_params.get("invert", False))
    elif src_choice == "Custom":
        palette_name = str(snaps.get("B", {}).get("params", {}).get("palette", "Inferno"))
        vmin = float(g.get("scale_custom_min", 0.0))
        vmax = float(g.get("scale_custom_max", 1.0))
        invert = False
    else:  # "Layer B"
        layer_params = snaps.get("B", {}).get("params", {})
        palette_name = str(layer_params.get("palette", "Inferno"))
        vmin = float(layer_params.get("vmin", 0.0))
        vmax = float(layer_params.get("vmax", 1000.0))
        invert = bool(layer_params.get("invert", False))

    mult = float(g.get("scale_mult", 0.001))
    if src_choice == "Custom":
        val_top = vmax
        val_bot = vmin
    else:
        val_top = vmax * mult
        val_bot = vmin * mult

    if invert:
        val_top, val_bot = val_bot, val_top

    val_mid = (val_top + val_bot) / 2.0

    diff = abs(val_top - val_bot)
    if diff < 0.005:
        fmt = "%.4f"
    elif diff < 0.5:
        fmt = "%.3f"
    elif diff < 50.0:
        fmt = "%.2f"
    else:
        fmt = "%.1f" if diff < 1000 else "%.0f"

    str_top = fmt % val_top
    str_mid = fmt % val_mid
    str_bot = fmt % val_bot

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, min(0.65, h / 1200.0 * 0.7))
    thickness = 1

    (w_top, h_top), _ = cv2.getTextSize(str_top, font, font_scale, thickness)
    (w_mid, h_mid), _ = cv2.getTextSize(str_mid, font, font_scale, thickness)
    (w_bot, h_bot), _ = cv2.getTextSize(str_bot, font, font_scale, thickness)
    max_text_w = max(w_top, w_mid, w_bot)
    max_text_h = max(h_top, h_mid, h_bot)

    bar_w = max(14, int(round(w * 0.02)))
    bar_w = min(bar_w, 28)
    bar_h = int(round(h * 0.55))
    bar_h = max(40, min(bar_h, h - 40))

    pad = 8
    panel_w = bar_w + max_text_w + pad * 3 + 4
    panel_h = bar_h + max_text_h + pad * 2

    pos = str(g.get("scale_pos", "Right")).lower()
    if pos == "left":
        panel_x0 = pad
    else:
        panel_x0 = max(0, w - panel_w - pad)
    panel_x1 = min(w, panel_x0 + panel_w)

    panel_y0 = max(0, (h - panel_h) // 2)
    panel_y1 = min(h, panel_y0 + panel_h)

    bgr_u8 = np.clip(bgr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    alpha_u8 = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # Semi-transparent dark backing panel for legibility
    bg_slice = bgr_u8[panel_y0:panel_y1, panel_x0:panel_x1].astype(np.float32)
    bgr_u8[panel_y0:panel_y1, panel_x0:panel_x1] = (bg_slice * 0.25).astype(np.uint8)
    alpha_u8[panel_y0:panel_y1, panel_x0:panel_x1] = np.maximum(
        alpha_u8[panel_y0:panel_y1, panel_x0:panel_x1], 230
    )
    cv2.rectangle(bgr_u8, (panel_x0, panel_y0), (panel_x1 - 1, panel_y1 - 1), (60, 60, 60), 1)

    bar_x0 = panel_x0 + pad
    bar_x1 = bar_x0 + bar_w
    bar_y0 = panel_y0 + pad + max_text_h // 2
    bar_y1 = bar_y0 + bar_h

    grad = np.linspace(255, 0, bar_h, dtype=np.uint8)
    if invert:
        grad = 255 - grad
    grad_color = palette_lut(palette_name)[grad]
    grad_tile = np.repeat(grad_color[:, None, :], bar_w, axis=1)

    bgr_u8[bar_y0:bar_y1, bar_x0:bar_x1] = grad_tile
    cv2.rectangle(bgr_u8, (bar_x0 - 1, bar_y0 - 1), (bar_x1, bar_y1), (220, 220, 220), 1)

    text_x = bar_x1 + pad
    tick_len = 4

    # Top
    y_top = bar_y0
    cv2.line(bgr_u8, (bar_x1, y_top), (bar_x1 + tick_len, y_top), (220, 220, 220), 1)
    cv2.putText(bgr_u8, str_top, (text_x, y_top + h_top // 2), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(bgr_u8, str_top, (text_x, y_top + h_top // 2), font, font_scale, (255, 255, 255), thickness)

    # Mid
    y_mid = (bar_y0 + bar_y1) // 2
    cv2.line(bgr_u8, (bar_x1, y_mid), (bar_x1 + tick_len, y_mid), (220, 220, 220), 1)
    cv2.putText(bgr_u8, str_mid, (text_x, y_mid + h_mid // 2), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(bgr_u8, str_mid, (text_x, y_mid + h_mid // 2), font, font_scale, (255, 255, 255), thickness)

    # Bot
    y_bot = bar_y1 - 1
    cv2.line(bgr_u8, (bar_x1, y_bot), (bar_x1 + tick_len, y_bot), (220, 220, 220), 1)
    cv2.putText(bgr_u8, str_bot, (text_x, y_bot + h_bot // 2), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(bgr_u8, str_bot, (text_x, y_bot + h_bot // 2), font, font_scale, (255, 255, 255), thickness)

    out_bgr = bgr_u8.astype(np.float32) * (1.0 / 255.0)
    out_a = alpha_u8.astype(np.float32) * (1.0 / 255.0)
    return out_bgr, out_a


def render_overlay(
    snaps: dict[str, dict[str, Any]],
    g: dict[str, Any],
    preview_max: int | None,
    export_scale: float | None = None,
    cache: dict[str, Any] | None = None,
) -> RenderResult:
    """
    Render both layers onto the canvas.

    ``preview_max`` limits the longest side of the rendered image (live preview);
    ``export_scale`` renders at a fixed scale instead (full resolution export).
    ``cache`` (optional dict) memoises the geometric warp per layer.
    """
    ox, oy, W, H = compute_canvas(snaps, g)
    if export_scale is not None:
        s = max(0.01, float(export_scale))
        if W * H * s * s > MAX_EXPORT_PIXELS:
            s = math.sqrt(MAX_EXPORT_PIXELS / float(W * H))
    else:
        s = min(1.0, float(preview_max or 1600) / float(max(W, H)))
    out_w = max(1, int(round(W * s)))
    out_h = max(1, int(round(H * s)))

    layers_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in ("A", "B"):
        sn = snaps[name]
        li: LayerImage | None = sn["image"]
        if li is None or not sn["params"]["visible"]:
            continue
        geo_key = tuple(sn["params"][k] for k in GEOMETRY_KEYS)
        key = (li.version, geo_key, sn["fit_H"].tobytes(), ox, oy, out_w, out_h, s)
        warped = None
        if cache is not None:
            hit = cache.get(name)
            if hit is not None and hit[0] == key:
                warped = hit[1]
        if warped is None:
            warped = warp_layer(li, sn["params"], sn["fit_H"], (ox, oy), out_w, out_h, s)
            if cache is not None:
                cache[name] = (key, warped)
        bgr, alpha, mono_mask, layer_mask = shade_layer(warped, li.kind, sn["params"])
        layers_data[name] = (bgr, alpha, mono_mask, layer_mask)

    # 1. Transfer Alpha (cross-layer alpha masking: e.g. A masks B while preserving B's cuts)
    transfer = str(g.get("transfer_alpha", "None"))
    if transfer == "Apply Alpha A -> B" and "A" in layers_data and "B" in layers_data:
        bgr_b, alpha_b, m_b, lmask_b = layers_data["B"]
        lmask_a = layers_data["A"][3]
        layers_data["B"] = (bgr_b, alpha_b * lmask_a, m_b, lmask_b * lmask_a)
    elif transfer == "Apply Alpha B -> A" and "A" in layers_data and "B" in layers_data:
        bgr_a, alpha_a, m_a, lmask_a = layers_data["A"]
        lmask_b = layers_data["B"][3]
        layers_data["A"] = (bgr_a, alpha_a * lmask_b, m_a, lmask_a * lmask_b)
    elif transfer == "Both (A * B)" and "A" in layers_data and "B" in layers_data:
        bgr_a, alpha_a, m_a, lmask_a = layers_data["A"]
        bgr_b, alpha_b, m_b, lmask_b = layers_data["B"]
        layers_data["A"] = (bgr_a, alpha_a * lmask_b, m_a, lmask_a * lmask_b)
        layers_data["B"] = (bgr_b, alpha_b * lmask_a, m_b, lmask_b * lmask_a)

    # 2. Processed as mask (mono intensity modulation)
    if g.get("processed_mask", False) and "B" in layers_data:
        src = str(g.get("mask_source", "Layer A"))
        mask: np.ndarray | None = None
        if src == "Layer A" and "A" in layers_data:
            mask = layers_data["A"][2]
        elif src == "Layer B":
            mask = layers_data["B"][2]
        elif src == "Both (A * B)" and "A" in layers_data:
            mask = layers_data["A"][2] * layers_data["B"][2]

        if mask is not None:
            bgr_b, alpha_b, m_b, lmask_b = layers_data["B"]
            layers_data["B"] = (bgr_b, alpha_b * mask, m_b, lmask_b * mask)

    layers: list[tuple[np.ndarray, np.ndarray]] = [
        (layers_data[name][0], layers_data[name][1]) for name in ("A", "B") if name in layers_data
    ]

    bgr, a = composite(layers, str(g["blend"]), (out_h, out_w))
    if g.get("show_scale", False):
        bgr, a = draw_intensity_scale(bgr, a, g, snaps)
    return RenderResult(bgr, a, W, H, ox, oy, s)


def render_raw(
    snap: dict[str, Any], g: dict[str, Any], processed: bool = False
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Render one layer in its own source space (no geometry) for viewing and point picking.

    If ``processed`` is True, applies colorization, vmin/vmax, palette, invert and threshold cuts.
    If ``processed`` is False, shows the raw sensor data (RGB natural colors or linear grayscale).
    """
    li: LayerImage | None = snap["image"]
    if li is None:
        return np.zeros((480, 640, 3), np.float32), np.zeros((480, 640), np.float32), 640, 480

    if processed:
        p = dict(snap["params"])
        p["opacity"] = 1.0
        bgr, a, _, _ = shade_layer(li.stack, li.kind, p)
        c, al = composite([(bgr, a)], "Normal", (li.h, li.w))
        return c, al, li.w, li.h

    if li.kind == "gray":
        cov = li.stack[:, :, 1]
        safe = np.where(cov > 1e-4, cov, 1.0)
        v = np.where(cov > 1e-4, li.stack[:, :, 0] / safe, 0.0).astype(np.float32)
        span = li.dmax - li.dmin
        if abs(span) < 1e-6:
            span = 1.0
        t = np.clip((v - li.dmin) / span, 0.0, 1.0)
        u8 = (t * 255.0 + 0.5).astype(np.uint8)
        bgr = np.ascontiguousarray(np.repeat(u8[:, :, None], 3, axis=2))
        a = np.where(cov > 1e-4, 1.0, 0.0).astype(np.float32)
    else:
        bgr = np.ascontiguousarray(li.stack[:, :, :3])
        a = (li.stack[:, :, 3].astype(np.float32) * (1.0 / 255.0))

    c, al = composite([(bgr, a)], "Normal", (li.h, li.w))
    return c, al, li.w, li.h


# ─────────────────────────────────────────────────────────────────────────────
# Point-based alignment
# ─────────────────────────────────────────────────────────────────────────────
def fit_transform(src: np.ndarray, dst: np.ndarray, model: str) -> tuple[np.ndarray | None, str, float, float]:
    """
    Least-squares fit of a 3x3 transform mapping ``src`` points onto ``dst`` points.

    Returns ``(H or None, message, mean_error, max_error)``.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    n = len(src)
    need = MIN_POINTS.get(model, 2)
    if n < need or len(dst) != n:
        return None, f"{model} needs at least {need} point pair(s) (have {n}).", 0.0, 0.0

    H = np.eye(3)
    if model == "Translation":
        t = (dst - src).mean(axis=0)
        H = translate(float(t[0]), float(t[1]))
    elif model == "Similarity":
        sc, dc = src.mean(axis=0), dst.mean(axis=0)
        zs = (src[:, 0] - sc[0]) + 1j * (src[:, 1] - sc[1])
        zd = (dst[:, 0] - dc[0]) + 1j * (dst[:, 1] - dc[1])
        den = float(np.sum(np.abs(zs) ** 2))
        if den < 1e-12:
            return None, "Degenerate points (all source points identical).", 0.0, 0.0
        a = np.sum(np.conj(zs) * zd) / den
        R = np.array([[a.real, -a.imag], [a.imag, a.real]])
        t = dc - R @ sc
        H = np.array([[R[0, 0], R[0, 1], t[0]], [R[1, 0], R[1, 1], t[1]], [0.0, 0.0, 1.0]])
    elif model == "Affine":
        X = np.hstack([src, np.ones((n, 1))])
        sol, *_ = np.linalg.lstsq(X, dst, rcond=None)
        H = np.vstack([sol.T, [0.0, 0.0, 1.0]])
    elif model == "Homography":
        Hh, _ = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32), 0)
        if Hh is None:
            return None, "Homography estimation failed (degenerate points).", 0.0, 0.0
        H = Hh.astype(np.float64)
    else:
        return None, f"Unknown model '{model}'.", 0.0, 0.0

    if not np.all(np.isfinite(H)) or abs(np.linalg.det(H[:2, :2])) < 1e-12:
        return None, "Fit produced a degenerate transform.", 0.0, 0.0

    err = np.linalg.norm(apply_h(H, src) - dst, axis=1)
    return H, "OK", float(err.mean()), float(err.max())


# ─────────────────────────────────────────────────────────────────────────────
# Settings (JSON) serialization
# ─────────────────────────────────────────────────────────────────────────────
def settings_to_dict(layers: dict[str, Layer], g: dict[str, Any]) -> dict[str, Any]:
    """Serialize all alignment settings to a JSON-compatible dict."""
    out: dict[str, Any] = {
        "type": "overlay_aligner_settings",
        "version": SETTINGS_VERSION,
        "global": {k: g[k] for k in default_global_params() if k in g},
        "layers": {},
    }
    for name, layer in layers.items():
        out["layers"][name] = {
            "params": {k: layer.params[k] for k in default_layer_params(name)},
            "fit_matrix": layer.fit_H.tolist(),
            "points": [[float(x), float(y)] for x, y in layer.points],
            "source_size": [layer.image.w, layer.image.h] if layer.image is not None else None,
        }
    return out


def apply_settings_dict(data: dict[str, Any], layers: dict[str, Layer], g: dict[str, Any]) -> list[str]:
    """
    Apply a settings dict (as produced by :func:`settings_to_dict`) in place.

    Unknown keys are ignored and invalid values fall back to defaults. Returns a list of
    human-readable warnings.
    """
    warnings: list[str] = []
    if not isinstance(data, dict) or data.get("type") != "overlay_aligner_settings":
        raise ValueError("Not an Overlay Aligner settings file.")
    if int(data.get("version", 0)) > SETTINGS_VERSION:
        warnings.append("Settings were written by a newer version; some values may be ignored.")

    gdefaults = default_global_params()
    for k, v in (data.get("global") or {}).items():
        if k in gdefaults:
            g[k] = _coerce(k, gdefaults[k], v)

    for name, ld in (data.get("layers") or {}).items():
        layer = layers.get(name)
        if layer is None or not isinstance(ld, dict):
            continue
        defaults = default_layer_params(name)
        for k, v in (ld.get("params") or {}).items():
            if k in defaults:
                layer.params[k] = _coerce(k, defaults[k], v)
        if "vmin" in (ld.get("params") or {}) or "vmax" in (ld.get("params") or {}):
            layer.range_user_set = True

        fm = ld.get("fit_matrix")
        try:
            arr = np.array(fm, dtype=np.float64)
            layer.fit_H = arr if arr.shape == (3, 3) and np.all(np.isfinite(arr)) else np.eye(3)
        except (TypeError, ValueError):
            layer.fit_H = np.eye(3)

        pts = []
        for pt in ld.get("points") or []:
            try:
                pts.append([float(pt[0]), float(pt[1])])
            except (TypeError, ValueError, IndexError):
                continue
        layer.points = pts

        ss = ld.get("source_size")
        if layer.image is not None and ss and list(ss) != [layer.image.w, layer.image.h]:
            warnings.append(
                f"Layer {name}: settings were made for a {ss[0]}x{ss[1]} image, "
                f"current image is {layer.image.w}x{layer.image.h}."
            )
    return warnings
