"""
Primary root generator for PyRootSim.

Generates synthetic primary (tap) root masks with configurable morphology.
The generation pipeline for a single root is:

1. **Path generation** — A stochastic angular noise signal (sum of
   frequency layers + optional kinks) is integrated to produce (x, y)
   coordinates along the root centreline.
2. **Width profiling** — A base width is sampled from a Beta distribution,
   then optionally modulated by taper, thin-variant scaling, pinch
   constrictions, and low-frequency fluctuation noise.
3. **Rendering** — The path is rasterised onto a binary mask using
   circles (for wide segments) and polylines (for thin segments).
4. **Artifact injection** — Edge-pixel dropout and dust pixels simulate
   realistic segmentation noise.
5. **Repair** — Disconnections caused by artifact dropout are surgically
   re-painted using the original path and width data.
6. **Cropping** — The mask is tight-cropped with padding, and an
   analytical skeleton + width profile are returned alongside metadata.

This module also exposes ``save_primary_root`` and
``generate_primary_root_dataset`` for batch generation to disk, but
the recommended entry point for programmatic use is
``generate_single_primary_root``.

Note
----
Several helper functions in this module (``crop_mask_tight``,
``keep_largest_component``, ``repair_disconnections``,
``generate_noise_signal``, ``apply_width_pinches``) are also used by
the lateral root generator. In a future refactor these will move to a
shared ``pyrootsim.utils`` module.
"""

import numpy as np
from PIL import Image
import os
import shutil
import json
from datetime import datetime
from tqdm import tqdm
import cv2
from scipy import ndimage

from pyrootsim.roots.configs import (
    BASE_FREQS,
    SMOOTH_FREQS,
    LONG_WAVE_FREQS,
    PRIMARY_CONFIGS,
    DEFAULT_PRIMARY_TAPER_CONFIG,
    DEFAULT_PRIMARY_PINCH_CONFIG,
    DEFAULT_PRIMARY_WIDTH_SKEW,
)


# =========================================================================
# Mask Helpers
# =========================================================================

def crop_mask_tight(mask, padding=10):
    """Crop a binary mask to its bounding box plus *padding* pixels.

    Args:
        mask:    2-D uint8 array (0 / 255).
        padding: pixels of margin around the non-zero region.

    Returns:
        (cropped_mask, x_offset, y_offset) where offsets record the
        top-left corner of the crop in the original coordinate frame.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask, 0, 0
    y_min = max(0, np.min(ys) - padding)
    y_max = min(mask.shape[0], np.max(ys) + padding + 1)
    x_min = max(0, np.min(xs) - padding)
    x_max = min(mask.shape[1], np.max(xs) + padding + 1)
    return mask[y_min:y_max, x_min:x_max], x_min, y_min


def get_bounding_box(mask):
    """Return ``{y_min, y_max, x_min, x_max}`` for the non-zero region,
    or ``None`` if the mask is empty."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return {
        "y_min": int(np.min(ys)),
        "y_max": int(np.max(ys)),
        "x_min": int(np.min(xs)),
        "x_max": int(np.max(xs)),
    }


def keep_largest_component(mask):
    """Keep only the largest 8-connected component in a binary mask.

    Args:
        mask: 2-D uint8 array (0 / 255).

    Returns:
        (filtered_mask, num_components) where *num_components* is the
        count before filtering.
    """
    binary_mask = (mask > 0).astype(np.uint8)
    labeled_array, num_components = ndimage.label(binary_mask)

    if num_components <= 1:
        return mask, num_components

    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0
    largest_label = np.argmax(component_sizes)

    filtered_mask = np.zeros_like(mask)
    filtered_mask[labeled_array == largest_label] = 255
    return filtered_mask, num_components


# =========================================================================
# Config Helpers
# =========================================================================

def _freq_layers_to_string(freq_layers):
    """Return a human-readable name for a frequency-layer preset."""
    if freq_layers == BASE_FREQS:
        return "BASE_FREQS"
    if freq_layers == SMOOTH_FREQS:
        return "SMOOTH_FREQS"
    if freq_layers == LONG_WAVE_FREQS:
        return "LONG_WAVE_FREQS"
    return "UNKNOWN"


def _get_taper_config(cfg):
    """Merge per-config taper overrides with the primary taper defaults."""
    tc = DEFAULT_PRIMARY_TAPER_CONFIG.copy()
    override = cfg.get("taper_override", None)
    if override:
        tc.update(override)
    return tc


def _get_pinch_config(cfg):
    """Merge per-config pinch overrides with the primary pinch defaults."""
    pc = DEFAULT_PRIMARY_PINCH_CONFIG.copy()
    override = cfg.get("pinch_override", None)
    if override:
        pc.update(override)
    return pc


def resolve_primary_config(config_name, rng):
    """Resolve a primary config name, handling mix-source configs.

    Mix-source configs (e.g. ``12_ExtraLong_Mixed_Sweeping``) randomly
    select one of their sub-configs. The returned dict includes private
    keys ``_resolved_from`` and ``_original_config`` for metadata.

    Args:
        config_name: key in :data:`PRIMARY_CONFIGS`.
        rng:         NumPy random generator.

    Returns:
        A resolved config dict (shallow copy, safe to mutate).
    """
    cfg = PRIMARY_CONFIGS[config_name]
    if "mix_sources" in cfg:
        source_key = rng.choice(cfg["mix_sources"])
        resolved = PRIMARY_CONFIGS[source_key].copy()
        resolved["_resolved_from"] = source_key
        resolved["_original_config"] = config_name
        return resolved
    cfg = cfg.copy()
    cfg["_resolved_from"] = config_name
    cfg["_original_config"] = config_name
    return cfg


# =========================================================================
# Noise & Path Generation
# =========================================================================

def generate_noise_signal(length, freq, amp, rng):
    """Generate a smoothed 1-D noise signal by interpolating random samples.

    Produces a signal of the given *length* by placing random control
    points at intervals of ``1 / freq``, interpolating linearly, and
    smoothing with a box kernel.

    Args:
        length: number of output samples.
        freq:   spatial frequency (controls spacing of control points).
        amp:    standard deviation of the Gaussian control-point values.
        rng:    NumPy random generator.

    Returns:
        1-D float array of shape ``(length,)``.
    """
    if length == 0:
        return np.array([])
    step = max(1, int(1 / freq))
    num_points = max(4, int(length / step) + 2)
    y_points = rng.normal(0, amp, num_points)
    x_points = np.arange(num_points) * step
    x = np.arange(length)
    y_interp = np.interp(x, x_points, y_points)
    kernel_size = max(3, step // 2)
    if kernel_size > 1:
        kernel = np.ones(kernel_size) / kernel_size
        y_interp = np.convolve(y_interp, kernel, mode="same")
    return y_interp[:length]


def generate_root_path(length, freq_layers, kink_prob, kink_amp, rng):
    """Generate a stochastic 2-D root centreline path.

    The path is built by summing angular noise layers (one per frequency
    preset), optionally injecting sharp kink events, and integrating the
    resulting angle signal into (x, y) coordinates via cumulative cosine
    and sine.

    The base direction is downward (π/2), so the root grows roughly
    top-to-bottom.

    Args:
        length:      number of path points (pixels along the centreline).
        freq_layers: list of ``(frequency, amplitude)`` tuples.
        kink_prob:   per-pixel probability of a kink event.
        kink_amp:    magnitude of each kink deviation (radians).
        rng:         NumPy random generator.

    Returns:
        ``(x_path, y_path)`` — two 1-D float arrays of shape ``(length,)``,
        both starting at 0.
    """
    amps = np.array([rng.normal(amp, amp * 0.2) for _, amp in freq_layers])
    angle_signal = np.zeros(length)
    for i, (freq, _) in enumerate(freq_layers):
        angle_signal += generate_noise_signal(length, freq, amps[i], rng)

    if kink_prob > 0:
        kink_mask = rng.random(length) < kink_prob
        kink_positions = np.where(kink_mask)[0]
        if len(kink_positions) > 0:
            kink_signal = np.zeros(length)
            directions = rng.choice([-1, 1], size=len(kink_positions))
            magnitudes = rng.normal(kink_amp, 0.1, size=len(kink_positions))
            durations = rng.integers(5, 15, size=len(kink_positions))
            for pos, d, m, dur in zip(
                kink_positions, directions, magnitudes, durations
            ):
                end = min(length, pos + dur)
                kink_signal[pos:end] += d * m
            angle_signal += kink_signal

    final_angles = angle_signal + (np.pi / 2)
    dx, dy = np.cos(final_angles), np.sin(final_angles)
    x, y = np.cumsum(dx), np.cumsum(dy)
    return x - x[0], y - y[0]


# =========================================================================
# Width Profile — Pinch System
# =========================================================================

def apply_width_pinches(width_profile, rng, pinch_cfg):
    """Apply stochastic localised constrictions to a width profile.

    Each pinch event is a symmetric dip centred at a random position
    along the root, with configurable depth and sharpness. A forbidden
    zone at both ends prevents pinches near the base or tip.

    The *width_profile* array is modified **in place**.

    Args:
        width_profile: 1-D float array of per-pixel widths (modified).
        rng:           NumPy random generator.
        pinch_cfg:     dict with keys from
                       :data:`DEFAULT_PRIMARY_PINCH_CONFIG`.

    Returns:
        List of pinch-event metadata dicts (empty if none applied).
    """
    length = len(width_profile)
    if length < 20:
        return []

    if rng.random() >= pinch_cfg["pinch_enabled_prob"]:
        return []

    min_c, max_c = pinch_cfg["count_range"]
    num_pinches = rng.integers(min_c, max_c + 1)

    start_forbidden = int(length * pinch_cfg["forbidden_start_pct"])
    end_forbidden = int(length * (1.0 - pinch_cfg["forbidden_end_pct"]))

    if end_forbidden <= start_forbidden + 10:
        return []

    pinch_events = []

    for _ in range(num_pinches):
        center = rng.integers(start_forbidden, end_forbidden)
        dur_frac = rng.uniform(*pinch_cfg["duration_frac_range"])
        half_dur = max(3, int(length * dur_frac / 2))
        depth = rng.uniform(*pinch_cfg["depth_factor_range"])
        sharpness = rng.uniform(*pinch_cfg["sharpness_range"])

        p_start = max(0, center - half_dur)
        p_end = min(length, center + half_dur)
        p_len = p_end - p_start

        if p_len < 3:
            continue

        t = np.linspace(0, 1, (p_len + 1) // 2 + 1)
        if p_len % 2 == 0:
            t_full = np.concatenate([t[:-1], t[::-1]])[:p_len]
        else:
            t_full = np.concatenate([t[:-1], t[-1:], t[-2::-1]])[:p_len]

        dip = 1.0 - (1.0 - depth) * (t_full ** sharpness)
        width_profile[p_start:p_end] *= dip[: p_end - p_start]

        pinch_events.append(
            {
                "center_idx": int(center),
                "half_duration": int(half_dur),
                "depth_factor": round(float(depth), 3),
                "sharpness": round(float(sharpness), 2),
            }
        )

    return pinch_events


# =========================================================================
# Width Profile — Full Generation (Beta-skewed + taper + thin + pinch)
# =========================================================================

def generate_width_profile(
    length, mean, std, rng, taper_cfg, pinch_cfg,
    width_skew=None, force_taper=None, force_thin=None,
):
    """Generate a complete width profile for a primary root.

    Sampling pipeline:

    1. Optionally apply thin-variant scaling (reduces effective mean).
    2. Sample a base width from a Beta distribution within
       ``[mean ± 2*std]``, clipped to ``[2.5, 12.0]``.
    3. Add low-frequency fluctuation noise.
    4. Apply taper (none / mild / strong) starting partway along the root.
    5. Apply stochastic pinch constrictions.
    6. Final clip to ``[1.0, 14.0]``.

    Args:
        length:      number of path points.
        mean:        configured mean width (pixels).
        std:         configured width standard deviation.
        rng:         NumPy random generator.
        taper_cfg:   taper configuration dict.
        pinch_cfg:   pinch configuration dict.
        width_skew:  ``(alpha, beta)`` for the Beta distribution, or
                     ``None`` to use :data:`DEFAULT_PRIMARY_WIDTH_SKEW`.
        force_taper: override taper bucket (``"none"``/``"mild"``/``"strong"``).
        force_thin:  override thin-variant flag.

    Returns:
        ``(width_profile, taper_bucket, is_thin, pinch_events, skew_info)``
    """
    if width_skew is None:
        width_skew = DEFAULT_PRIMARY_WIDTH_SKEW

    alpha, beta_param = width_skew

    # --- Thin variant ---
    if force_thin is not None:
        is_thin = force_thin
    else:
        is_thin = rng.random() < taper_cfg["thin_prob"]

    effective_mean = mean
    if is_thin:
        scale = rng.uniform(*taper_cfg["thin_scale_range"])
        effective_mean = max(2.5, mean * scale)

    # --- Beta-distributed base width ---
    w_min = max(2.5, effective_mean - 2.0 * std)
    w_max = min(12.0, effective_mean + 2.0 * std)

    beta_sample = rng.beta(alpha, beta_param)
    base = w_min + beta_sample * (w_max - w_min)
    base = np.clip(base, 2.5, 12.0)

    skew_info = {
        "alpha": float(alpha),
        "beta": float(beta_param),
        "beta_sample": round(float(beta_sample), 4),
        "w_min": round(float(w_min), 2),
        "w_max": round(float(w_max), 2),
        "base_width": round(float(base), 2),
    }

    # --- Fluctuation noise ---
    fluctuation = generate_noise_signal(length, 0.05, std * 0.3, rng)
    width_profile = np.clip(base + fluctuation, 2.5, 14)

    # --- Taper ---
    if force_taper is not None:
        taper_bucket = force_taper
    else:
        roll = rng.random()
        p_none = taper_cfg["no_taper_prob"]
        p_mild = taper_cfg["mild_taper_prob"]
        if roll < p_none:
            taper_bucket = "none"
        elif roll < p_none + p_mild:
            taper_bucket = "mild"
        else:
            taper_bucket = "strong"

    if taper_bucket in ("mild", "strong"):
        if taper_bucket == "mild":
            taper_start = rng.uniform(*taper_cfg["mild_taper_start_range"])
            taper_end_frac = rng.uniform(
                *taper_cfg["mild_taper_end_fraction_range"]
            )
        else:
            taper_start = rng.uniform(*taper_cfg["strong_taper_start_range"])
            taper_end_frac = rng.uniform(
                *taper_cfg["strong_taper_end_fraction_range"]
            )
        start_idx = int(length * taper_start)
        taper_len = length - start_idx
        if taper_len > 1:
            t = np.linspace(0, 1, taper_len)
            decay = 1.0 - (1.0 - taper_end_frac) * (t ** 1.8)
            width_profile[start_idx:] *= decay

    # --- Pinch ---
    pinch_events = apply_width_pinches(width_profile, rng, pinch_cfg)
    width_profile = np.clip(width_profile, 1.0, 14.0)

    return width_profile, taper_bucket, is_thin, pinch_events, skew_info


# =========================================================================
# Mask Rendering
# =========================================================================

def render_root_mask(height, width, x_path, y_path, width_profile,
                     x_offset=None, y_offset=10):
    """Rasterise a root path with variable width onto a binary mask.

    Wide segments (width ≥ 2 px) are drawn as filled circles at each path
    point. Thin segments (width < 2 px) are drawn as a 1-px polyline.
    All drawing uses ``cv2.LINE_8`` (no anti-aliasing) to produce clean
    binary output.

    Args:
        height, width: canvas dimensions.
        x_path, y_path: root centreline coordinates (float arrays).
        width_profile:   per-point diameter (float array, same length).
        x_offset:        horizontal offset applied to *x_path* (defaults
                         to centering the path on the canvas).
        y_offset:        vertical offset applied to *y_path*.

    Returns:
        2-D uint8 array of shape ``(height, width)`` with values 0 / 255.
    """
    canvas = np.zeros((height, width), dtype=np.uint8)
    if x_offset is None:
        x_offset = width // 2 - int(x_path[0])

    px = (x_path + x_offset).astype(np.int32)
    py = (y_path + y_offset).astype(np.int32)

    n = len(px)
    i = 0
    while i < n:
        if width_profile[i] < 2.0:
            seg_start = i
            while i < n and width_profile[i] < 2.0:
                i += 1
            if i - seg_start >= 2:
                pts = np.column_stack([px[seg_start:i], py[seg_start:i]])
                cv2.polylines(
                    canvas, [pts], isClosed=False, color=255,
                    thickness=1, lineType=cv2.LINE_8,
                )
            elif i - seg_start == 1:
                cx, cy = int(px[seg_start]), int(py[seg_start])
                if 0 <= cx < width and 0 <= cy < height:
                    canvas[cy, cx] = 255
        else:
            cx, cy = int(px[i]), int(py[i])
            if 0 <= cx < width and 0 <= cy < height:
                r = max(1, int(round(width_profile[i] / 2.0)))
                cv2.circle(canvas, (cx, cy), r, 255, -1, lineType=cv2.LINE_8)
            i += 1

    return canvas


# =========================================================================
# Artifact Injection
# =========================================================================

def apply_artifacts(mask, mode, rng):
    """Simulate segmentation noise on a binary root mask.

    Two effects are applied:

    * **Edge dropout** — boundary pixels are randomly removed.
    * **Dust** — isolated pixels are randomly added near the root edges.

    The *mode* controls severity: ``"none"`` skips artifacts entirely,
    ``"low"`` applies light noise, ``"standard"`` moderate noise, and
    ``"high"`` aggressive noise.

    Args:
        mask: 2-D uint8 array (0 / 255). Not modified in place.
        mode: one of ``"none"``, ``"low"``, ``"standard"``, ``"high"``.
        rng:  NumPy random generator.

    Returns:
        New 2-D uint8 array with artifacts applied.
    """
    if mode == "none":
        return (mask > 0).astype(np.uint8) * 255

    binary = (mask > 0).astype(np.uint8) * 255
    drop_prob = {"standard": 0.10, "low": 0.03, "high": 0.30}.get(mode, 0.10)
    dust_prob = {"standard": 0.02, "low": 0.005, "high": 0.10}.get(mode, 0.02)

    kernel = np.ones((3, 3), np.uint8)

    # Edge dropout
    eroded = cv2.erode(binary, kernel, iterations=1)
    edge_mask = (binary > 0) & (eroded == 0)
    edge_indices = np.where(edge_mask)
    if len(edge_indices[0]) > 0:
        drop_mask = rng.random(len(edge_indices[0])) < drop_prob
        binary[edge_indices[0][drop_mask], edge_indices[1][drop_mask]] = 0

    # Dust addition
    dilated = cv2.dilate(binary, kernel, iterations=1)
    dust_zone = (dilated > 0) & (binary == 0)
    dust_indices = np.where(dust_zone)
    if len(dust_indices[0]) > 0:
        dust_add = rng.random(len(dust_indices[0])) < dust_prob
        binary[dust_indices[0][dust_add], dust_indices[1][dust_add]] = 255

    return binary


# =========================================================================
# Connectivity Repair
# =========================================================================

def repair_disconnections(mask, x_path, y_path, width_profile,
                          x_offset, y_offset):
    """Repair disconnections in a rendered root mask.

    Operates in two phases:

    **Phase 1 — Re-paint damaged path pixels.**  Compares the current
    mask against the path coordinates. Any path index whose pixel is zero
    gets re-rendered using the original width profile, plus a small
    neighbourhood margin.

    **Phase 2 — Bridge remaining components.**  If multiple 8-connected
    components remain after phase 1, the closest boundary-pixel pairs
    between components are connected with lines whose thickness matches
    the local width profile. Repeats until fully connected (up to 20
    iterations).

    The *mask* array is modified **in place**.

    Args:
        mask:          binary mask (H, W), uint8 0/255.
        x_path:        root path x-coordinates (float array).
        y_path:        root path y-coordinates (float array).
        width_profile: width at each path index (float array).
        x_offset:      x offset used during rendering.
        y_offset:      y offset used during rendering.

    Returns:
        ``(mask, num_repaired)`` — the repaired mask and a count of
        path indices re-painted plus bridges drawn.
    """
    h, w = mask.shape
    px = (x_path + x_offset).astype(np.int32)
    py = (y_path + y_offset).astype(np.int32)
    n = len(px)
    total_repaired = 0

    # --- Phase 1: Re-paint damaged path centre pixels ---
    damaged_indices = []
    for idx in range(n):
        cx, cy = int(px[idx]), int(py[idx])
        if 0 <= cx < w and 0 <= cy < h:
            if mask[cy, cx] == 0:
                damaged_indices.append(idx)

    if damaged_indices:
        damaged_set = set(damaged_indices)
        margin = 2
        for idx in damaged_indices:
            for offset in range(-margin, margin + 1):
                neighbor = idx + offset
                if 0 <= neighbor < n:
                    damaged_set.add(neighbor)

        for idx in sorted(damaged_set):
            cx, cy = int(px[idx]), int(py[idx])
            if not (0 <= cx < w and 0 <= cy < h):
                continue

            wp = width_profile[idx]
            if wp < 2.0:
                mask[cy, cx] = 255
                next_idx = idx + 1
                if next_idx in damaged_set and next_idx < n:
                    nx, ny = int(px[next_idx]), int(py[next_idx])
                    if 0 <= nx < w and 0 <= ny < h:
                        cv2.line(
                            mask, (cx, cy), (nx, ny), 255,
                            thickness=1, lineType=cv2.LINE_8,
                        )
            else:
                r = max(1, int(round(wp / 2.0)))
                cv2.circle(mask, (cx, cy), r, 255, -1, lineType=cv2.LINE_8)

        total_repaired += len(damaged_indices)

    # --- Phase 2: Bridge remaining disconnected components ---
    struct_8 = np.ones((3, 3), dtype=int)
    labeled, num_cc = ndimage.label(mask > 0, structure=struct_8)

    max_bridge_iterations = 20
    iteration = 0

    while num_cc > 1 and iteration < max_bridge_iterations:
        iteration += 1

        comp_sizes = np.bincount(labeled.ravel())
        comp_sizes[0] = 0
        largest_label = np.argmax(comp_sizes)

        # Boundary pixels of the largest component
        largest_mask_arr = (labeled == largest_label).astype(np.uint8)
        k = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(largest_mask_arr, k, iterations=1)
        largest_boundary = (largest_mask_arr > 0) & (eroded == 0)
        largest_pts = np.argwhere(largest_boundary)
        if len(largest_pts) == 0:
            largest_pts = np.argwhere(labeled == largest_label)

        bridged_any = False

        for comp_label in range(1, num_cc + 1):
            if comp_label == largest_label:
                continue

            comp_mask_arr = (labeled == comp_label).astype(np.uint8)
            eroded_comp = cv2.erode(comp_mask_arr, k, iterations=1)
            comp_boundary = (comp_mask_arr > 0) & (eroded_comp == 0)
            comp_pts = np.argwhere(comp_boundary)
            if len(comp_pts) == 0:
                comp_pts = np.argwhere(labeled == comp_label)
            if len(comp_pts) == 0:
                continue

            # Subsample large boundary sets for speed
            max_pts = 500
            lp_sub = largest_pts
            if len(largest_pts) > max_pts:
                idx_sub = np.linspace(
                    0, len(largest_pts) - 1, max_pts, dtype=int
                )
                lp_sub = largest_pts[idx_sub]

            cp_sub = comp_pts
            if len(comp_pts) > max_pts:
                idx_sub = np.linspace(
                    0, len(comp_pts) - 1, max_pts, dtype=int
                )
                cp_sub = comp_pts[idx_sub]

            # Find closest boundary-pixel pair (chunked to limit memory)
            best_dist = np.inf
            best_lp = best_cp = None
            chunk = 200

            for ci in range(0, len(cp_sub), chunk):
                cp_chunk = cp_sub[ci : ci + chunk]
                dy = cp_chunk[:, 0:1] - lp_sub[:, 0].reshape(1, -1)
                dx = cp_chunk[:, 1:2] - lp_sub[:, 1].reshape(1, -1)
                dists = dy * dy + dx * dx
                min_idx = np.unravel_index(np.argmin(dists), dists.shape)
                min_dist = dists[min_idx]
                if min_dist < best_dist:
                    best_dist = min_dist
                    best_cp = cp_chunk[min_idx[0]]
                    best_lp = lp_sub[min_idx[1]]

            if best_lp is None or best_cp is None:
                continue

            # Bridge thickness from width profile at bridge midpoint
            mid_y = (best_lp[0] + best_cp[0]) / 2.0
            mid_x = (best_lp[1] + best_cp[1]) / 2.0
            path_dists = (
                (py.astype(float) - mid_y) ** 2
                + (px.astype(float) - mid_x) ** 2
            )
            nearest_path_idx = np.argmin(path_dists)
            bridge_width = width_profile[nearest_path_idx]
            bridge_thickness = max(1, int(round(bridge_width / 2.0)))

            pt1 = (int(best_lp[1]), int(best_lp[0]))
            pt2 = (int(best_cp[1]), int(best_cp[0]))
            cv2.line(
                mask, pt1, pt2, 255,
                thickness=bridge_thickness, lineType=cv2.LINE_8,
            )

            total_repaired += 1
            bridged_any = True

        if not bridged_any:
            break

        labeled, num_cc = ndimage.label(mask > 0, structure=struct_8)

    return mask, total_repaired


# =========================================================================
# Analytical Skeleton
# =========================================================================

def _get_analytical_skeleton(x_path, y_path, x_offset, y_offset):
    """Convert root path coordinates to skeleton points in mask space.

    Returns an ``(N, 2)`` array of ``(row, col)`` — i.e. ``(y, x)`` — in
    the coordinate frame of the rendered mask.
    """
    skeleton_x = x_path + x_offset
    skeleton_y = y_path + y_offset
    return np.column_stack([skeleton_y, skeleton_x])


# =========================================================================
# Single Root Generation (Main Entry Point)
# =========================================================================

def generate_single_primary_root(config_name, root_id, seed,
                                 force_taper=None, force_thin=None):
    """Generate a single primary root mask with full metadata.

    This is the main entry point for producing one primary root. It runs
    the full pipeline: config resolution → path generation → width
    profiling → rendering → artifact injection → repair → cropping.

    Args:
        config_name: key in :data:`PRIMARY_CONFIGS`.
        root_id:     integer index for this root (used in metadata).
        seed:        random seed for full reproducibility.
        force_taper: override taper bucket (``"none"``/``"mild"``/``"strong"``).
        force_thin:  override thin-variant flag.

    Returns:
        A dict with keys ``mask``, ``label_img``, ``skeleton_points``,
        ``skeleton_widths``, and ``metadata``; or ``None`` if generation
        failed (e.g. empty mask after filtering).
    """
    rng = np.random.default_rng(seed)

    cfg = resolve_primary_config(config_name, rng)
    taper_cfg = _get_taper_config(cfg)
    pinch_cfg = _get_pinch_config(cfg)

    r_len = int(rng.normal(cfg["len_mean"], cfg["len_std"]))
    r_len = max(50, r_len)

    x_path, y_path = generate_root_path(
        r_len, cfg["freq_layers"], cfg["kink_prob"], cfg["kink_amp"], rng,
    )

    width_skew = cfg.get("width_skew", DEFAULT_PRIMARY_WIDTH_SKEW)
    width_profile, taper_bucket, is_thin, pinch_events, skew_info = (
        generate_width_profile(
            r_len, cfg["width_mean"], cfg["width_std"], rng,
            taper_cfg, pinch_cfg,
            width_skew=width_skew,
            force_taper=force_taper, force_thin=force_thin,
        )
    )

    # Render onto a temporary canvas large enough to hold the root
    temp_h = r_len + 200
    temp_w = 800
    x_offset = temp_w // 2 - int(x_path[0])
    y_offset = 10

    raw_mask = render_root_mask(
        temp_h, temp_w, x_path, y_path, width_profile, x_offset, y_offset,
    )

    # Artifact injection
    artifact_mask = apply_artifacts(raw_mask, cfg["artifact_lvl"], rng)

    # Repair disconnections caused by artifacts
    artifact_mask, num_repaired = repair_disconnections(
        artifact_mask, x_path, y_path, width_profile, x_offset, y_offset,
    )

    # Keep largest component as final safety net
    filtered_mask, num_components = keep_largest_component(artifact_mask)
    if num_components == 0:
        return None

    # Crop
    cropped_mask, offset_x, offset_y = crop_mask_tight(
        filtered_mask, padding=15,
    )
    if np.sum(cropped_mask > 0) == 0:
        return None

    # Analytical skeleton in cropped coordinates
    skeleton_points = _get_analytical_skeleton(
        x_path, y_path, x_offset - offset_x, y_offset - offset_y,
    )
    valid = (
        (skeleton_points[:, 0] >= 0)
        & (skeleton_points[:, 0] < cropped_mask.shape[0])
        & (skeleton_points[:, 1] >= 0)
        & (skeleton_points[:, 1] < cropped_mask.shape[1])
    )
    skeleton_points = skeleton_points[valid]
    skeleton_widths = width_profile[valid]

    if skeleton_points is None or len(skeleton_points) < 10:
        return None

    # Label image (trivial for a single primary: all foreground = 1)
    label_img = np.zeros(cropped_mask.shape, dtype=np.uint16)
    label_img[cropped_mask > 0] = 1

    bbox = get_bounding_box(cropped_mask)

    metadata = {
        "root_id": root_id,
        "config_source": cfg["_original_config"],
        "resolved_config": cfg["_resolved_from"],
        "category": cfg["category"],
        "generation_params": {
            "len_mean": cfg["len_mean"],
            "len_std": cfg["len_std"],
            "freq_layers": _freq_layers_to_string(cfg["freq_layers"]),
            "kink_prob": cfg["kink_prob"],
            "kink_amp": cfg["kink_amp"],
            "artifact_lvl": cfg["artifact_lvl"],
            "width_mean": cfg["width_mean"],
            "width_std": cfg["width_std"],
        },
        "width_info": {
            "taper_bucket": taper_bucket,
            "is_thin": is_thin,
            "num_pinch_events": len(pinch_events),
            "pinch_events": pinch_events,
            "skew_info": skew_info,
        },
        "actual_length_px": int(len(skeleton_points)),
        "requested_length_px": r_len,
        "bounding_box": bbox,
        "skeleton_length": int(len(skeleton_points)),
        "label_id": 1,
        "crop_offset": {"x": int(offset_x), "y": int(offset_y)},
        "connected_components": {
            "num_found": int(num_components),
            "largest_kept": num_components > 1,
        },
        "repair_info": {
            "num_path_indices_repaired": num_repaired,
        },
        "seed": seed,
        "generation_timestamp": datetime.now().isoformat(),
    }

    return {
        "mask": cropped_mask,
        "label_img": label_img,
        "skeleton_points": skeleton_points,
        "skeleton_widths": skeleton_widths,
        "metadata": metadata,
    }


# =========================================================================
# Disk I/O
# =========================================================================

def save_primary_root(output_dir, root_data, config_name, root_idx):
    """Save a generated primary root to disk.

    Creates four files inside ``output_dir/config_name/``:

    * ``{root_id}_primary_mask.png`` — binary mask
    * ``{root_id}_primary_labels.png`` — label image
    * ``{root_id}_skeleton.npy`` — skeleton coordinates
    * ``{root_id}_widths.npy`` — per-point width profile
    * ``{root_id}_primary.json`` — full metadata

    Args:
        output_dir:  base output directory.
        root_data:   dict returned by :func:`generate_single_primary_root`.
        config_name: primary config key (used as subdirectory name).
        root_idx:    integer index (used to build the root ID).

    Returns:
        The root ID string.
    """
    root_id = f"{config_name}_{root_idx:05d}"
    root_dir = os.path.join(output_dir, config_name)
    os.makedirs(root_dir, exist_ok=True)

    mask_path = os.path.join(root_dir, f"{root_id}_primary_mask.png")
    Image.fromarray(root_data["mask"]).save(mask_path)

    label_path = os.path.join(root_dir, f"{root_id}_primary_labels.png")
    Image.fromarray(root_data["label_img"]).save(label_path)

    skeleton_path = os.path.join(root_dir, f"{root_id}_skeleton.npy")
    np.save(skeleton_path, root_data["skeleton_points"])

    widths_path = os.path.join(root_dir, f"{root_id}_widths.npy")
    np.save(widths_path, root_data["skeleton_widths"])

    root_data["metadata"]["root_id"] = root_id
    root_data["metadata"]["files"] = {
        "mask": f"{root_id}_primary_mask.png",
        "labels": f"{root_id}_primary_labels.png",
        "skeleton": f"{root_id}_skeleton.npy",
        "widths": f"{root_id}_widths.npy",
    }

    json_path = os.path.join(root_dir, f"{root_id}_primary.json")
    with open(json_path, "w") as f:
        json.dump(root_data["metadata"], f, indent=2)

    return root_id


# =========================================================================
# Batch Dataset Generation
# =========================================================================

def generate_primary_root_dataset(
    output_dir="primary_roots_dataset",
    num_per_config=100,
    configs_to_generate=None,
    clear_existing=True,
    base_seed=42,
):
    """Generate a full primary-root dataset across multiple configs.

    For each config, generates *num_per_config* roots, saving masks,
    labels, skeletons, width profiles, and per-root JSON metadata. A
    ``generation_summary.json`` is written to *output_dir* at the end.

    Args:
        output_dir:         base output directory.
        num_per_config:     number of roots per config.
        configs_to_generate: list of config keys, or ``None`` for all.
        clear_existing:     if ``True``, remove *output_dir* first.
        base_seed:          base random seed for reproducibility.

    Returns:
        Summary dict with generation statistics.
    """
    if clear_existing and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if configs_to_generate is None:
        configs_to_generate = sorted(PRIMARY_CONFIGS.keys())

    summary = {
        "output_dir": output_dir,
        "num_per_config": num_per_config,
        "base_seed": base_seed,
        "generator_version": "pyrootsim",
        "configs": {},
        "total_generated": 0,
        "total_failed": 0,
        "total_multi_component": 0,
        "taper_stats": {"none": 0, "mild": 0, "strong": 0},
        "thin_count": 0,
        "pinch_stats": {"roots_with_pinches": 0, "total_pinch_events": 0},
        "repair_stats": {"roots_repaired": 0, "total_indices_repaired": 0},
        "generation_timestamp": datetime.now().isoformat(),
    }

    print(f"Generating {num_per_config} roots per config...")
    print(f"Output directory: {output_dir}")
    print(f"Configs: {len(configs_to_generate)}")
    print("-" * 50)

    for config_name in configs_to_generate:
        print(f"\nProcessing: {config_name}")
        generated = 0
        failed = 0
        multi_component_count = 0
        config_taper_stats = {"none": 0, "mild": 0, "strong": 0}
        config_thin_count = 0
        config_pinch_roots = 0
        config_pinch_events = 0
        config_repaired_roots = 0
        config_repaired_indices = 0

        pbar = tqdm(total=num_per_config, desc=config_name, leave=True)

        attempts = 0
        max_attempts = num_per_config * 3

        while generated < num_per_config and attempts < max_attempts:
            seed = base_seed + attempts + (hash(config_name) % 100000)
            attempts += 1

            root_data = generate_single_primary_root(
                config_name, generated, seed,
            )

            if root_data is None:
                failed += 1
                continue

            if root_data["metadata"]["connected_components"]["largest_kept"]:
                multi_component_count += 1

            wi = root_data["metadata"]["width_info"]
            config_taper_stats[wi["taper_bucket"]] += 1
            if wi["is_thin"]:
                config_thin_count += 1
            if wi["num_pinch_events"] > 0:
                config_pinch_roots += 1
                config_pinch_events += wi["num_pinch_events"]

            ri = root_data["metadata"]["repair_info"]
            if ri["num_path_indices_repaired"] > 0:
                config_repaired_roots += 1
                config_repaired_indices += ri["num_path_indices_repaired"]

            save_primary_root(output_dir, root_data, config_name, generated)
            generated += 1
            pbar.update(1)

        pbar.close()

        summary["configs"][config_name] = {
            "generated": generated,
            "failed": failed,
            "attempts": attempts,
            "multi_component_filtered": multi_component_count,
            "taper_stats": config_taper_stats,
            "thin_count": config_thin_count,
            "pinch_roots": config_pinch_roots,
            "pinch_events_total": config_pinch_events,
            "repaired_roots": config_repaired_roots,
            "repaired_indices_total": config_repaired_indices,
        }
        summary["total_generated"] += generated
        summary["total_failed"] += failed
        summary["total_multi_component"] += multi_component_count
        for k in config_taper_stats:
            summary["taper_stats"][k] += config_taper_stats[k]
        summary["thin_count"] += config_thin_count
        summary["pinch_stats"]["roots_with_pinches"] += config_pinch_roots
        summary["pinch_stats"]["total_pinch_events"] += config_pinch_events
        summary["repair_stats"]["roots_repaired"] += config_repaired_roots
        summary["repair_stats"]["total_indices_repaired"] += (
            config_repaired_indices
        )

        print(
            f"  Generated: {generated}, Failed: {failed}, "
            f"Taper: {config_taper_stats}, Thin: {config_thin_count}, "
            f"Pinched: {config_pinch_roots}/{generated}, "
            f"Repaired: {config_repaired_roots}/{generated}"
        )

    summary_path = os.path.join(output_dir, "generation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 50)
    print(f"COMPLETE: {summary['total_generated']} roots generated")
    print(f"Taper distribution: {summary['taper_stats']}")
    print(f"Thin variants: {summary['thin_count']}")
    print(f"Pinch stats: {summary['pinch_stats']}")
    print(f"Repair stats: {summary['repair_stats']}")
    print(f"Multi-component cases handled: {summary['total_multi_component']}")
    print(f"Summary saved to: {summary_path}")

    return summary