"""
Lateral root generator for PyRootSim.

Generates lateral (branch) roots attached to an existing primary root.
The pipeline for a single primary+laterals composite is:

1. **Style resolution** — The primary config name is mapped to a lateral
   curvature/artifact style.
2. **Lateral count & placement** — The number of laterals is sampled,
   sides (left/right) are assigned according to a stochastic pattern,
   and attachment points are placed along the primary skeleton respecting
   emergence-zone weights and minimum-spacing constraints.
3. **Top-tip laterals** — Optional extra laterals at or near the primary
   tip are sampled independently.
4. **Per-lateral generation** — Each lateral gets its own stochastic
   path (angular noise + drift), width profile (Beta-sampled start width,
   taper, pinch), and is rendered with the "continuous spine" method
   (1 px polyline backbone + circle overlays for thickness).
5. **Repair & filtering** — Each lateral mask is repaired for
   disconnections, then the component attached to the anchor point is
   kept (discarding floating fragments).
6. **Compositing** — All lateral masks are composited onto an expanded
   canvas containing the primary mask, producing a combined mask and a
   label image.
7. **Connectivity retry** — If the composite is not 8-connected, the
   entire attempt is re-run with a new seed (up to
   :data:`MAX_COMPOSITE_RETRIES` times).

Two public entry points are provided:

* :func:`generate_lateral_root_inline` — takes a primary root dict
  (as returned by the primary generator) and produces the composite
  entirely in memory. **Recommended for programmatic use.**
* :func:`process_single_root_from_disk` — loads a primary root from
  disk (JSON + files) and processes it. Useful for batch pipelines that
  save primaries first.

Note
----
Several helper functions in this module (``generate_noise_signal``,
``apply_width_pinches``, ``keep_largest_component``,
``repair_disconnections``, ``check_cc8``) are duplicated from
``pyrootsim.roots.primary``. In a future refactor these will move to a
shared ``pyrootsim.utils`` module.
"""

import numpy as np
import cv2
from PIL import Image
import os
import json
from scipy import ndimage

from pyrootsim.roots.configs import (
    LATERAL_STYLE_MAPPING,
    LATERAL_MODES,
    LATERAL_EMERGENCE_ZONE,
    LATERAL_LR_PATTERNS,
    LATERAL_WIDTH_CONFIG,
    LATERAL_SPACING_CONFIG,
    DEFAULT_LATERAL_TAPER_CONFIG,
    LATERAL_TAPER_DISABLED_CONFIGS,
    DEFAULT_LATERAL_PINCH_CONFIG,
    LATERAL_PINCH_DISABLED_CONFIGS,
    DEFAULT_LATERAL_WIDTH_SKEW,
    LATERAL_WIDTH_SKEW_DISABLED_CONFIGS,
    TOP_TIP_CONFIG,
    TOP_TIP_DISABLED_CONFIGS,
    MAX_COMPOSITE_RETRIES,
)

# Re-use shared helpers from primary (will move to utils/ in a future refactor)
from pyrootsim.roots.primary import (
    generate_noise_signal,
    apply_width_pinches,
    keep_largest_component,
    repair_disconnections,
)


# =========================================================================
# Skeleton Helpers
# =========================================================================

def _ensure_skeleton_top_to_bottom(skeleton_points, skeleton_widths):
    """Flip skeleton arrays so that the first point is at the top (lower y).

    The primary generator may produce skeletons in either direction; this
    normalises them so index 0 is the root base (top of the image) and
    index -1 is the tip (bottom).
    """
    if len(skeleton_points) < 2:
        return skeleton_points, skeleton_widths
    if skeleton_points[0][0] > skeleton_points[-1][0]:
        skeleton_points = skeleton_points[::-1].copy()
        skeleton_widths = skeleton_widths[::-1].copy()
    return skeleton_points, skeleton_widths


def _estimate_angle_at_point(skeleton_points, idx, window=10):
    """Estimate the local tangent angle of the skeleton at *idx*.

    Uses a symmetric window around *idx* and returns the angle in radians
    (``atan2(dy, dx)`` convention, where y points downward).
    """
    start_idx = max(0, idx - window)
    end_idx = min(len(skeleton_points) - 1, idx + window)
    p1, p2 = skeleton_points[start_idx], skeleton_points[end_idx]
    return np.arctan2(p2[0] - p1[0], p2[1] - p1[1])


# =========================================================================
# Component Helpers
# =========================================================================

def _keep_attached_component(mask, anchor_y, anchor_x):
    """Keep only the 8-connected component that contains the anchor point.

    Unlike :func:`keep_largest_component`, this preserves the component
    touching the attachment point even if it is not the largest — this
    prevents floating tail fragments from surviving when a lateral is
    severed by artifact dropout.

    Falls back to :func:`keep_largest_component` if the anchor does not
    touch any foreground pixel.
    """
    binary_mask = (mask > 0).astype(np.uint8)
    labeled_array, num_components = ndimage.label(
        binary_mask, structure=np.ones((3, 3), dtype=int),
    )

    if num_components <= 1:
        return mask, num_components

    h, w = mask.shape
    ay, ax = int(anchor_y), int(anchor_x)

    target_label = 0
    for radius in range(2):
        y_min = max(0, ay - radius)
        y_max = min(h, ay + radius + 1)
        x_min = max(0, ax - radius)
        x_max = min(w, ax + radius + 1)

        patch = labeled_array[y_min:y_max, x_min:x_max]
        labels_in_patch = patch[patch > 0]
        if len(labels_in_patch) > 0:
            target_label = np.bincount(labels_in_patch).argmax()
            break

    if target_label == 0:
        return keep_largest_component(mask)

    filtered_mask = np.zeros_like(mask)
    filtered_mask[labeled_array == target_label] = 255
    return filtered_mask, num_components


def _check_cc8(mask):
    """Return the number of 8-connected components in a binary mask."""
    binary = (mask > 0).astype(np.uint8)
    _, n = ndimage.label(binary, structure=np.ones((3, 3), dtype=int))
    return n


# =========================================================================
# Style Resolution
# =========================================================================

def _resolve_lateral_style(config_name, rng):
    """Resolve the lateral curvature/artifact style for a primary config.

    Mix-source configs (12_, 13_, 14_, 15_) are randomly mapped to one
    of their constituent sub-configs before lookup.

    Returns:
        ``(style_dict, resolved_config_name)``
    """
    resolved = config_name
    if config_name.startswith(("12_", "12a_", "12b_")):
        resolved = rng.choice(["12a_ExtraLong_Gentle", "12b_ExtraLong_Sweeping"])
    elif config_name.startswith("13_"):
        resolved = rng.choice([
            "04_Medium_Kinky_Noisy", "05_Medium_Smooth_Snake",
            "06_Medium_Clean_GroundTruth",
        ])
    elif config_name.startswith("14_"):
        resolved = rng.choice([
            "07_Long_Kinky_Noisy", "08_Long_Sweeping_Curves",
            "09_Long_Smooth_Static",
        ])
    elif config_name.startswith("15_"):
        resolved = rng.choice(["10_ExtraLong_Hybrid", "11_ExtraLong_Curvy_Clean"])

    fallback = LATERAL_STYLE_MAPPING["06_Medium_Clean_GroundTruth"]
    return LATERAL_STYLE_MAPPING.get(resolved, fallback), resolved


# =========================================================================
# Lateral Count & Placement Sampling
# =========================================================================

def _sample_lateral_count(lateral_mode, rng):
    """Sample the number of laterals from a Beta-shaped distribution."""
    min_c, max_c = lateral_mode["count_range"]
    if min_c >= max_c:
        return min_c
    return max(
        min_c,
        min(max_c, int(min_c + rng.beta(2, 3) * (max_c - min_c + 1))),
    )


def _sample_lr_pattern(rng):
    """Sample a left/right distribution pattern."""
    probs = np.array([p["prob"] for p in LATERAL_LR_PATTERNS])
    probs /= probs.sum()
    return LATERAL_LR_PATTERNS[rng.choice(len(LATERAL_LR_PATTERNS), p=probs)]


def _assign_sides(num_laterals, pattern, rng):
    """Assign each lateral to ``"left"`` or ``"right"`` based on *pattern*."""
    if num_laterals == 0:
        return []
    if pattern["name"] == "all_one_side":
        side = "left" if rng.random() < 0.5 else "right"
        return [side] * num_laterals
    if pattern["alternating"]:
        sides = []
        current = "left" if rng.random() < 0.5 else "right"
        for _ in range(num_laterals):
            sides.append(current)
            if rng.random() < 0.70:
                current = "right" if current == "left" else "left"
        return sides
    l_frac = rng.uniform(*pattern["left_frac"])
    num_left = int(round(num_laterals * l_frac))
    sides = ["left"] * num_left + ["right"] * (num_laterals - num_left)
    rng.shuffle(sides)
    return list(sides)


def _sample_attachment_points(num_laterals, skeleton_length, sides, rng):
    """Sample skeleton indices where laterals will attach.

    Uses weighted emergence zones and enforces minimum spacing between
    same-side and opposite-side attachment points.
    """
    if num_laterals == 0:
        return []
    forbidden = 1.0 - LATERAL_EMERGENCE_ZONE["forbidden_tip_pct"]
    min_same = int(skeleton_length * LATERAL_SPACING_CONFIG["min_same_side_pct"])
    min_opp = int(skeleton_length * LATERAL_SPACING_CONFIG["min_opposite_side_pct"])
    points, l_pts, r_pts = [], [], []

    for _ in range(num_laterals * 30):
        if len(points) == num_laterals:
            break
        zones = LATERAL_EMERGENCE_ZONE["zones"]
        w = np.array([z["weight"] for z in zones])
        w /= w.sum()
        pct = rng.uniform(*zones[rng.choice(len(zones), p=w)]["range"])
        if pct > forbidden:
            continue
        idx = int(pct * skeleton_length)
        side = sides[len(points)]
        same_list = l_pts if side == "left" else r_pts
        opp_list = r_pts if side == "left" else l_pts
        if (any(abs(idx - p) < min_same for p in same_list)
                and rng.random() > LATERAL_SPACING_CONFIG["overlap_prob"]):
            continue
        if any(abs(idx - p) < min_opp for p in opp_list) and rng.random() > 0.80:
            continue
        points.append(idx)
        if side == "left":
            l_pts.append(idx)
        else:
            r_pts.append(idx)
    return points


# =========================================================================
# Lateral Path Generation
# =========================================================================

def _generate_lateral_path(length, primary_angle, side, style, lateral_mode, rng):
    """Generate a stochastic 2-D path for a normal lateral root.

    The initial angle deviates from the primary tangent by a random offset
    (whose sign depends on *side*), then angular noise and gravitropic
    drift are applied.
    """
    sign = 1 if side == "left" else -1
    init_angle = primary_angle + rng.uniform(*lateral_mode["angle_offset_range"]) * sign
    angle_signal = np.zeros(length)

    for freq, amp in style["freq_layers"]:
        sampled_amp = max(0.001, rng.normal(amp, amp * 0.2))
        angle_signal += generate_noise_signal(length, freq, sampled_amp, rng)

    if style["kink_prob"] > 0:
        k_mask = rng.random(length) < style["kink_prob"]
        for pos in np.where(k_mask)[0]:
            end = min(length, pos + rng.integers(5, 15))
            angle_signal[pos:end] += (
                rng.choice([-1, 1]) * rng.normal(style["kink_amp"], 0.1)
            )

    drift = np.linspace(0, rng.uniform(*lateral_mode["drift_range"]), length)
    angle_signal += drift * (-sign)

    final_angles = np.clip(init_angle + angle_signal, -np.pi / 4, 5 * np.pi / 4)
    return np.cumsum(np.cos(final_angles)), np.cumsum(np.sin(final_angles))


# =========================================================================
# Top-Tip Lateral Helpers
# =========================================================================

def _sample_near_tip_index_biased(skeleton_length, rng, max_pct=0.06):
    """Sample a skeleton index near the tip with geometric decay."""
    near_end = max(1, int(skeleton_length * max_pct))
    near_end = min(near_end, skeleton_length - 1)
    if near_end <= 1:
        return 1
    p = 0.55
    k = 1
    while k < near_end and rng.random() > p:
        k += 1
    return k


def _generate_top_tip_path(length, primary_angle, side, style, rng):
    """Generate a path for a top-tip lateral, optionally with an upward kick."""
    tt = TOP_TIP_CONFIG
    sign = 1 if side == "left" else -1
    init_angle = primary_angle + rng.uniform(*tt["angle_offset_range"]) * sign
    angle_signal = np.zeros(length)

    # Optional upward kick at the start
    if rng.random() < tt["upward_kick_prob"]:
        kick_frac = rng.uniform(*tt["upward_kick_length_frac"])
        kick_len = max(3, int(length * kick_frac))
        kick_boost = rng.uniform(*tt["upward_kick_angle_boost"])
        rise = np.linspace(0, kick_boost, kick_len // 2 + 1)
        fall = np.linspace(kick_boost, 0, kick_len - kick_len // 2)
        kick_curve = np.concatenate([rise[:-1], fall])[:kick_len]
        kick_profile = np.zeros(length)
        kick_profile[:kick_len] = kick_curve * sign
        angle_signal += kick_profile

    for freq, amp in style["freq_layers"]:
        sampled_amp = max(0.001, rng.normal(amp, amp * 0.2))
        angle_signal += generate_noise_signal(length, freq, sampled_amp, rng)

    if style["kink_prob"] > 0:
        k_mask = rng.random(length) < style["kink_prob"]
        for pos in np.where(k_mask)[0]:
            end = min(length, pos + rng.integers(5, 15))
            angle_signal[pos:end] += (
                rng.choice([-1, 1]) * rng.normal(style["kink_amp"], 0.1)
            )

    drift = np.linspace(0, rng.uniform(*tt["drift_range"]), length)
    angle_signal += drift * (-sign)

    final_angles = np.clip(init_angle + angle_signal, -np.pi / 4, 5 * np.pi / 4)
    return np.cumsum(np.cos(final_angles)), np.cumsum(np.sin(final_angles))


def _sample_top_tip_count(rng):
    """Sample how many top-tip laterals to inject (0, 1, or 2)."""
    tt = TOP_TIP_CONFIG
    if rng.random() >= tt["injection_prob"]:
        return 0
    counts = sorted(tt["count_weights"].keys())
    probs = np.array([tt["count_weights"][c] for c in counts])
    probs /= probs.sum()
    return int(rng.choice(counts, p=probs))


def _sample_top_tip_near_extras(rng):
    """Sample extra near-tip laterals (only if top-tip laterals exist)."""
    tt = TOP_TIP_CONFIG
    if rng.random() >= tt["near_tip_extra_prob"]:
        return 0
    counts = sorted(tt["near_tip_extra_count"].keys())
    probs = np.array([tt["near_tip_extra_count"][c] for c in counts])
    probs /= probs.sum()
    return int(rng.choice(counts, p=probs))


def _assign_top_tip_sides(count, rng):
    """Assign left/right sides for top-tip laterals."""
    if count == 0:
        return []
    if count == 1:
        return ["left" if rng.random() < 0.5 else "right"]
    first = "left" if rng.random() < 0.5 else "right"
    if rng.random() < TOP_TIP_CONFIG["opposite_side_prob"]:
        second = "right" if first == "left" else "left"
    else:
        second = first
    return [first, second]


def _sample_top_tip_attachments(num_tip, num_near, skeleton_length, rng):
    """Sample attachment indices for top-tip and near-tip laterals.

    Returns a sorted list of ``(skeleton_index, is_near_tip)`` tuples.
    """
    attachments = []
    for _ in range(num_tip):
        attachments.append((0, False))
    for _ in range(num_near):
        idx = _sample_near_tip_index_biased(
            skeleton_length, rng,
            max_pct=TOP_TIP_CONFIG["attachment_zone_near_pct"][1],
        )
        attachments.append((idx, True))
    attachments.sort(key=lambda x: x[0])
    return attachments


# =========================================================================
# Lateral Width Profiles
# =========================================================================

def _generate_lateral_width_profile(
    lat_len, parent_width, rng, config_name,
    primary_is_thin=False, force_taper=None, force_thin=None,
):
    """Generate a width profile for a normal lateral root.

    The start width is a fraction of the parent primary width, sampled
    via a Beta distribution (or uniform for clean configs). Taper, thin
    scaling, and pinch are applied similarly to the primary generator.
    An attachment zone at the base enforces a minimum width near the
    junction.

    Returns:
        ``(width_profile, taper_bucket, is_thin, pinch_events, skew_info)``
    """
    tc = DEFAULT_LATERAL_TAPER_CONFIG.copy()
    if config_name in LATERAL_TAPER_DISABLED_CONFIGS:
        tc.update({
            "no_taper_prob": 1.0, "mild_taper_prob": 0.0,
            "strong_taper_prob": 0.0, "thin_prob_normal_parent": 0.0,
            "thin_prob_thin_parent": 0.0,
        })

    # --- Start width sampling ---
    sf_min, sf_max = LATERAL_WIDTH_CONFIG["start_fraction_range"]
    if config_name in LATERAL_WIDTH_SKEW_DISABLED_CONFIGS:
        s_frac = rng.uniform(sf_min, sf_max)
        skew_info = {
            "alpha": 1.0, "beta": 1.0,
            "beta_sample": round(float((s_frac - sf_min) / (sf_max - sf_min)), 4),
        }
    else:
        alpha, beta_param = DEFAULT_LATERAL_WIDTH_SKEW
        beta_sample = rng.beta(alpha, beta_param)
        s_frac = sf_min + beta_sample * (sf_max - sf_min)
        skew_info = {
            "alpha": float(alpha), "beta": float(beta_param),
            "beta_sample": round(float(beta_sample), 4),
        }

    sw = max(2.5, parent_width * s_frac)
    skew_info["start_fraction"] = round(float(s_frac), 4)
    skew_info["start_width"] = round(float(sw), 2)

    # --- Thin variant ---
    if force_thin is not None:
        is_thin = force_thin
    else:
        thin_prob = (
            tc["thin_prob_thin_parent"] if primary_is_thin
            else tc["thin_prob_normal_parent"]
        )
        is_thin = rng.random() < thin_prob

    if is_thin:
        sw = max(2.5, sw * rng.uniform(*tc["thin_scale_range"]))

    # --- Base profile ---
    ew = max(2.5, sw * LATERAL_WIDTH_CONFIG["end_fraction_of_start"])
    base_profile = np.linspace(sw, ew, lat_len)
    noise = generate_noise_signal(lat_len, 0.08, 0.10, rng)
    width_profile = np.clip(base_profile + noise, 2.5, sw)

    # --- Taper ---
    if force_taper is not None:
        taper_bucket = force_taper
    else:
        roll = rng.random()
        if roll < tc["no_taper_prob"]:
            taper_bucket = "none"
        elif roll < tc["no_taper_prob"] + tc["mild_taper_prob"]:
            taper_bucket = "mild"
        else:
            taper_bucket = "strong"

    min_w = LATERAL_WIDTH_CONFIG["min_width_pixels_default"]
    if taper_bucket in ("mild", "strong"):
        if taper_bucket == "mild":
            taper_start = rng.uniform(*tc["mild_taper_start_range"])
            taper_end_frac = rng.uniform(*tc["mild_taper_end_fraction_range"])
        else:
            taper_start = rng.uniform(*tc["strong_taper_start_range"])
            taper_end_frac = rng.uniform(*tc["strong_taper_end_fraction_range"])
        start_idx = int(lat_len * taper_start)
        taper_length = lat_len - start_idx
        if taper_length > 1:
            t = np.linspace(0, 1, taper_length)
            decay = 1.0 - (1.0 - taper_end_frac) * (t ** 1.8)
            width_profile[start_idx:] *= decay
        min_w = LATERAL_WIDTH_CONFIG["min_width_pixels_tapered"]

    # --- Pinch ---
    pinch_cfg = DEFAULT_LATERAL_PINCH_CONFIG
    if config_name in LATERAL_PINCH_DISABLED_CONFIGS:
        pinch_cfg = DEFAULT_LATERAL_PINCH_CONFIG.copy()
        pinch_cfg["pinch_enabled_prob"] = 0.0
    pinch_events = apply_width_pinches(width_profile, rng, pinch_cfg)

    width_profile = np.clip(width_profile, min_w, sw)

    # --- Attachment zone (enforce minimum width near base) ---
    az_len = min(tc["attachment_zone_pixels"], lat_len)
    az_min = sw * tc["attachment_zone_min_fraction"]
    if az_len > 0:
        width_profile[:az_len] = np.maximum(width_profile[:az_len], az_min)

    return width_profile, taper_bucket, is_thin, pinch_events, skew_info


def _generate_top_tip_width_profile(
    lat_len, parent_width, rng, config_name,
    primary_is_thin=False, force_taper=None, force_thin=None,
):
    """Generate a width profile for a top-tip lateral.

    Identical to :func:`_generate_lateral_width_profile` except the start
    fraction range comes from :data:`TOP_TIP_CONFIG` instead of
    :data:`LATERAL_WIDTH_CONFIG`.
    """
    tc = DEFAULT_LATERAL_TAPER_CONFIG.copy()
    if config_name in LATERAL_TAPER_DISABLED_CONFIGS:
        tc.update({
            "no_taper_prob": 1.0, "mild_taper_prob": 0.0,
            "strong_taper_prob": 0.0, "thin_prob_normal_parent": 0.0,
            "thin_prob_thin_parent": 0.0,
        })

    sf_min, sf_max = TOP_TIP_CONFIG["width_start_fraction_range"]
    if config_name in LATERAL_WIDTH_SKEW_DISABLED_CONFIGS:
        s_frac = rng.uniform(sf_min, sf_max)
        skew_info = {
            "alpha": 1.0, "beta": 1.0,
            "beta_sample": round(
                float((s_frac - sf_min) / max(0.001, sf_max - sf_min)), 4
            ),
        }
    else:
        alpha, beta_param = DEFAULT_LATERAL_WIDTH_SKEW
        beta_sample = rng.beta(alpha, beta_param)
        s_frac = sf_min + beta_sample * (sf_max - sf_min)
        skew_info = {
            "alpha": float(alpha), "beta": float(beta_param),
            "beta_sample": round(float(beta_sample), 4),
        }

    sw = max(2.5, parent_width * s_frac)
    skew_info["start_fraction"] = round(float(s_frac), 4)
    skew_info["start_width"] = round(float(sw), 2)

    if force_thin is not None:
        is_thin = force_thin
    else:
        thin_prob = (
            tc["thin_prob_thin_parent"] if primary_is_thin
            else tc["thin_prob_normal_parent"]
        )
        is_thin = rng.random() < thin_prob

    if is_thin:
        sw = max(2.5, sw * rng.uniform(*tc["thin_scale_range"]))

    min_w_default = LATERAL_WIDTH_CONFIG["min_width_pixels_default"]
    ew = max(min_w_default, sw * LATERAL_WIDTH_CONFIG["end_fraction_of_start"])
    base_profile = np.linspace(sw, ew, lat_len)
    noise = generate_noise_signal(lat_len, 0.08, 0.10, rng)
    width_profile = np.clip(base_profile + noise, min_w_default, sw)

    if force_taper is not None:
        taper_bucket = force_taper
    else:
        roll = rng.random()
        if roll < tc["no_taper_prob"]:
            taper_bucket = "none"
        elif roll < tc["no_taper_prob"] + tc["mild_taper_prob"]:
            taper_bucket = "mild"
        else:
            taper_bucket = "strong"

    min_w = min_w_default
    if taper_bucket in ("mild", "strong"):
        if taper_bucket == "mild":
            taper_start = rng.uniform(*tc["mild_taper_start_range"])
            taper_end_frac = rng.uniform(*tc["mild_taper_end_fraction_range"])
        else:
            taper_start = rng.uniform(*tc["strong_taper_start_range"])
            taper_end_frac = rng.uniform(*tc["strong_taper_end_fraction_range"])
        start_idx = int(lat_len * taper_start)
        taper_length = lat_len - start_idx
        if taper_length > 1:
            t = np.linspace(0, 1, taper_length)
            decay = 1.0 - (1.0 - taper_end_frac) * (t ** 1.8)
            width_profile[start_idx:] *= decay
        min_w = LATERAL_WIDTH_CONFIG["min_width_pixels_tapered"]

    pinch_cfg = DEFAULT_LATERAL_PINCH_CONFIG
    if config_name in LATERAL_PINCH_DISABLED_CONFIGS:
        pinch_cfg = DEFAULT_LATERAL_PINCH_CONFIG.copy()
        pinch_cfg["pinch_enabled_prob"] = 0.0
    pinch_events = apply_width_pinches(width_profile, rng, pinch_cfg)

    width_profile = np.clip(width_profile, min_w, sw)

    az_len = min(tc["attachment_zone_pixels"], lat_len)
    az_min = sw * tc["attachment_zone_min_fraction"]
    if az_len > 0:
        width_profile[:az_len] = np.maximum(width_profile[:az_len], az_min)

    return width_profile, taper_bucket, is_thin, pinch_events, skew_info


# =========================================================================
# Lateral Rendering & Compositing
# =========================================================================

def _render_lateral_mask(x_path, y_path, widths, local_h, local_w,
                         offset_x, offset_y):
    """Render a lateral root onto a local canvas using the continuous-spine method.

    First draws an unbroken 1 px polyline backbone across the entire path,
    then overlays filled circles at points where width ≥ 2 px. This
    guarantees gap-free rendering without post-hoc repair.
    """
    canvas = np.zeros((local_h, local_w), dtype=np.uint8)
    px = (x_path + offset_x).astype(np.int32)
    py = (y_path + offset_y).astype(np.int32)

    # 1. Continuous 1 px spine
    pts = np.column_stack([px, py])
    cv2.polylines(
        canvas, [pts], isClosed=False, color=255,
        thickness=1, lineType=cv2.LINE_8,
    )

    # 2. Circle overlays for thickness
    for i in range(len(px)):
        if widths[i] >= 2.0:
            cx, cy = int(px[i]), int(py[i])
            if 0 <= cx < local_w and 0 <= cy < local_h:
                r = max(1, int(round(widths[i] / 2.0)))
                cv2.circle(canvas, (cx, cy), r, 255, -1, lineType=cv2.LINE_8)

    return canvas


def _compute_lateral_bbox(x_path, y_path, widths, attach_y, attach_x,
                          padding=5):
    """Compute the bounding box for a lateral in absolute coordinates.

    Returns ``(x_min, x_max, y_min, y_max)`` with width-based margin.
    """
    abs_x = x_path + attach_x
    abs_y = y_path + attach_y
    mw = int(np.ceil(np.max(widths)))
    return (
        int(np.floor(np.min(abs_x))) - mw - padding,
        int(np.ceil(np.max(abs_x))) + mw + padding,
        int(np.floor(np.min(abs_y))) - mw - padding,
        int(np.ceil(np.max(abs_y))) + mw + padding,
    )


def _composite_to_canvas(canvas, local_mask, x_min, y_min):
    """Composite a local mask onto a larger canvas using ``max`` blending.

    Handles out-of-bounds clipping automatically.
    """
    lh, lw = local_mask.shape
    ch, cw = canvas.shape
    sy, sx = max(0, -y_min), max(0, -x_min)
    dy, dx = max(0, y_min), max(0, x_min)
    ey, ex = min(lh, ch - y_min), min(lw, cw - x_min)
    if ey > sy and ex > sx:
        canvas[dy : dy + (ey - sy), dx : dx + (ex - sx)] = np.maximum(
            canvas[dy : dy + (ey - sy), dx : dx + (ex - sx)],
            local_mask[sy:ey, sx:ex],
        )
    return canvas


# =========================================================================
# Single-Attempt Lateral Assembly (Internal)
# =========================================================================

def _generate_laterals_single_attempt(
    primary_mask_orig, skeleton_points_orig, skeleton_widths,
    root_id, config_name, category, primary_is_thin,
    lateral_mode_name, lateral_mode, seed,
    force_taper=None, force_thin=None,
):
    """Run one attempt at generating and compositing all laterals.

    Returns ``(result_dict, cc8_count)`` where *cc8_count* is the number
    of 8-connected components in the final composite (1 = success).
    """
    rng = np.random.default_rng(seed)

    skeleton_points_orig, skeleton_widths = _ensure_skeleton_top_to_bottom(
        skeleton_points_orig, skeleton_widths,
    )
    skeleton_length = len(skeleton_points_orig)

    style, _ = _resolve_lateral_style(config_name, rng)
    num_laterals = _sample_lateral_count(lateral_mode, rng)

    if config_name in TOP_TIP_DISABLED_CONFIGS:
        num_top_tip = 0
        num_near_tip = 0
    else:
        num_top_tip = _sample_top_tip_count(rng)
        num_near_tip = _sample_top_tip_near_extras(rng) if num_top_tip > 0 else 0

    orig_h, orig_w = primary_mask_orig.shape

    # --- Zero-lateral shortcut ---
    if num_laterals == 0 and num_top_tip == 0 and num_near_tip == 0:
        padding = 50
        new_h, new_w = orig_h + 2 * padding, orig_w + 2 * padding
        combined_mask = np.zeros((new_h, new_w), dtype=np.uint8)
        combined_mask[padding : padding + orig_h, padding : padding + orig_w] = (
            primary_mask_orig
        )
        label_img = np.zeros((new_h, new_w), dtype=np.uint16)
        label_img[combined_mask > 0] = 1
        result = {
            "root_id": root_id,
            "combined_mask": combined_mask,
            "label_img": label_img,
            "primary_mask": combined_mask.copy(),
            "metadata": {
                "root_id": root_id,
                "lateral_mode": lateral_mode_name,
                "num_laterals": 0,
                "num_top_tip_laterals": 0,
                "cc_filtered_count": 0,
                "composite_cc8_retries": 0,
                "lateral_repairs": 0,
                "lateral_repair_indices_total": 0,
                "description": lateral_mode.get("description", ""),
            },
        }
        return result, 1

    # --- Sample normal laterals ---
    pattern = _sample_lr_pattern(rng)
    sides = _assign_sides(num_laterals, pattern, rng)
    attachment_indices = _sample_attachment_points(
        num_laterals, skeleton_length, sides, rng,
    )
    if len(attachment_indices) < len(sides):
        sides = sides[: len(attachment_indices)]

    # --- Sample top-tip laterals ---
    total_tt = num_top_tip + num_near_tip
    tt_sides = _assign_top_tip_sides(total_tt, rng)
    tt_attachments = _sample_top_tip_attachments(
        num_top_tip, num_near_tip, skeleton_length, rng,
    )

    # --- Build per-lateral data ---
    lateral_data = []
    min_x, max_x, min_y, max_y = 0, orig_w, 0, orig_h

    # Normal laterals
    for at_idx, side in zip(attachment_indices, sides):
        at_pt = skeleton_points_orig[at_idx]
        p_angle = _estimate_angle_at_point(skeleton_points_orig, at_idx)
        p_width = skeleton_widths[at_idx]

        length_frac = rng.uniform(*lateral_mode["length_range"])
        taper_factor = 1.0 if at_idx / skeleton_length <= 0.5 else 0.4
        lat_len = max(15, int(skeleton_length * length_frac * taper_factor))

        xp, yp = _generate_lateral_path(
            lat_len, p_angle, side, style, lateral_mode, rng,
        )
        widths, taper_bucket, is_thin, pinch_events, lat_skew = (
            _generate_lateral_width_profile(
                lat_len, p_width, rng, config_name,
                primary_is_thin=primary_is_thin,
                force_taper=force_taper, force_thin=force_thin,
            )
        )

        abs_x, abs_y = xp + at_pt[1], yp + at_pt[0]
        mw = int(np.ceil(np.max(widths)))
        min_x = min(min_x, int(np.floor(np.min(abs_x))) - mw)
        max_x = max(max_x, int(np.ceil(np.max(abs_x))) + mw)
        min_y = min(min_y, int(np.floor(np.min(abs_y))) - mw)
        max_y = max(max_y, int(np.ceil(np.max(abs_y))) + mw)

        lateral_data.append({
            "xp": xp, "yp": yp, "widths": widths,
            "at_pt": at_pt, "at_idx": at_idx, "side": side,
            "length": lat_len, "p_angle": p_angle,
            "taper_bucket": taper_bucket, "is_thin": is_thin,
            "pinch_events": pinch_events, "skew_info": lat_skew,
            "is_top_tip": False,
        })

    # Top-tip laterals
    for (at_idx, is_near), side in zip(tt_attachments, tt_sides):
        at_pt = skeleton_points_orig[at_idx]
        p_angle = _estimate_angle_at_point(skeleton_points_orig, at_idx)
        p_width = skeleton_widths[at_idx]

        length_frac = rng.uniform(*TOP_TIP_CONFIG["length_range"])
        lat_len = max(15, int(skeleton_length * length_frac))

        xp, yp = _generate_top_tip_path(lat_len, p_angle, side, style, rng)
        widths, taper_bucket, is_thin, pinch_events, tt_skew = (
            _generate_top_tip_width_profile(
                lat_len, p_width, rng, config_name,
                primary_is_thin=primary_is_thin,
                force_taper=force_taper, force_thin=force_thin,
            )
        )

        abs_x, abs_y = xp + at_pt[1], yp + at_pt[0]
        mw = int(np.ceil(np.max(widths)))
        min_x = min(min_x, int(np.floor(np.min(abs_x))) - mw)
        max_x = max(max_x, int(np.ceil(np.max(abs_x))) + mw)
        min_y = min(min_y, int(np.floor(np.min(abs_y))) - mw)
        max_y = max(max_y, int(np.ceil(np.max(abs_y))) + mw)

        lateral_data.append({
            "xp": xp, "yp": yp, "widths": widths,
            "at_pt": at_pt, "at_idx": at_idx, "side": side,
            "length": lat_len, "p_angle": p_angle,
            "taper_bucket": taper_bucket, "is_thin": is_thin,
            "pinch_events": pinch_events, "skew_info": tt_skew,
            "is_top_tip": True, "is_near_tip": is_near,
        })

    # --- Expanded canvas ---
    padding = 50
    ex_l = max(0, -min_x) + padding
    ex_t = max(0, -min_y) + padding
    new_h = orig_h + ex_t + max(0, max_y - orig_h) + padding
    new_w = orig_w + ex_l + max(0, max_x - orig_w) + padding

    primary_mask = np.zeros((new_h, new_w), dtype=np.uint8)
    primary_mask[ex_t : ex_t + orig_h, ex_l : ex_l + orig_w] = primary_mask_orig
    combined_mask = primary_mask.copy()
    label_img = np.zeros((new_h, new_w), dtype=np.uint16)
    label_img[primary_mask > 0] = 1

    laterals_meta = []
    lateral_masks = []
    cc_filtered_count = 0
    total_lateral_repairs = 0
    total_lateral_repair_indices = 0

    for i, lat in enumerate(lateral_data):
        # Anchor inside the primary mask at the exact skeleton coordinate
        ay = lat["at_pt"][0] + ex_t
        ax = lat["at_pt"][1] + ex_l

        xmi, xma, ymi, yma = _compute_lateral_bbox(
            lat["xp"], lat["yp"], lat["widths"], ay, ax,
        )

        l_mask = _render_lateral_mask(
            lat["xp"], lat["yp"], lat["widths"],
            yma - ymi, xma - xmi, ax - xmi, ay - ymi,
        )
        l_mask = (l_mask > 0).astype(np.uint8) * 255

        # Repair disconnections (mostly a safety net with the continuous renderer)
        l_mask, num_repaired = repair_disconnections(
            l_mask, lat["xp"], lat["yp"], lat["widths"], ax - xmi, ay - ymi,
        )
        if num_repaired > 0:
            total_lateral_repairs += 1
            total_lateral_repair_indices += num_repaired

        # Keep the component attached to the anchor point
        local_anchor_y = ay - ymi
        local_anchor_x = ax - xmi
        l_mask_filtered, num_comp = _keep_attached_component(
            l_mask, local_anchor_y, local_anchor_x,
        )
        if num_comp > 1:
            cc_filtered_count += 1

        full_lat_mask = np.zeros((new_h, new_w), dtype=np.uint8)
        _composite_to_canvas(full_lat_mask, l_mask_filtered, xmi, ymi)
        _composite_to_canvas(combined_mask, full_lat_mask, 0, 0)

        lateral_masks.append(full_lat_mask)
        laterals_meta.append({
            "lateral_id": i + 1,
            "side": lat["side"],
            "attachment_point": [int(ay), int(ax)],
            "attachment_skeleton_idx": int(lat["at_idx"]),
            "length_px": int(lat["length"]),
            "label_id": i + 2,
            "cc_filtered": num_comp > 1,
            "taper_bucket": lat["taper_bucket"],
            "is_thin": lat["is_thin"],
            "num_pinch_events": len(lat["pinch_events"]),
            "pinch_events": lat["pinch_events"],
            "skew_info": lat.get("skew_info", {}),
            "is_top_tip": lat["is_top_tip"],
            "is_near_tip": lat.get("is_near_tip", False),
            "num_repaired_indices": num_repaired,
        })

    # Assign label IDs to laterals
    for i, lm in enumerate(lateral_masks):
        label_img[(lm > 0) & (label_img == 0)] = i + 2

    cc8_count = _check_cc8(combined_mask)
    num_tt_actual = sum(1 for m in laterals_meta if m["is_top_tip"])

    result = {
        "root_id": root_id,
        "combined_mask": combined_mask,
        "label_img": label_img,
        "primary_mask": primary_mask,
        "metadata": {
            "root_id": root_id,
            "config_source": config_name,
            "lateral_mode": lateral_mode_name,
            "num_laterals": len(laterals_meta),
            "num_normal_laterals": len(laterals_meta) - num_tt_actual,
            "num_top_tip_laterals": num_tt_actual,
            "cc_filtered_count": cc_filtered_count,
            "lateral_repairs": total_lateral_repairs,
            "lateral_repair_indices_total": total_lateral_repair_indices,
            "description": lateral_mode.get("description", ""),
            "laterals": laterals_meta,
            "seed": seed,
        },
    }
    return result, cc8_count


# =========================================================================
# Public Entry Points
# =========================================================================

def process_single_root_from_disk(primary_json_path, root_idx,
                                  total_in_config, seed):
    """Load a primary root from disk and generate laterals for it.

    The lateral mode is deterministically assigned based on *root_idx*
    and *total_in_config* so that all modes are evenly covered across a
    batch.

    Args:
        primary_json_path: path to the primary root's JSON metadata file.
        root_idx:          index of this root within its config batch.
        total_in_config:   total number of roots in the config batch.
        seed:              random seed.

    Returns:
        Composite result dict (same structure as
        :func:`generate_lateral_root_inline`).
    """
    with open(primary_json_path, "r") as f:
        primary_meta = json.load(f)

    root_dir = os.path.dirname(primary_json_path)
    root_id = primary_meta["root_id"]
    config_name = primary_meta["config_source"]
    category = primary_meta["category"]
    primary_is_thin = primary_meta.get("width_info", {}).get("is_thin", False)

    modes_dict = LATERAL_MODES[category]
    mode_keys = sorted(modes_dict.keys())
    bucket_size = total_in_config // len(mode_keys)
    mode_name = mode_keys[min(root_idx // bucket_size, len(mode_keys) - 1)]
    lateral_mode = modes_dict[mode_name]

    primary_mask_orig = np.array(
        Image.open(os.path.join(root_dir, primary_meta["files"]["mask"]))
    )
    skeleton_points_orig = np.load(
        os.path.join(root_dir, primary_meta["files"]["skeleton"])
    )
    skeleton_widths = np.load(
        os.path.join(root_dir, primary_meta["files"]["widths"])
    )

    for attempt in range(MAX_COMPOSITE_RETRIES):
        result, cc8_count = _generate_laterals_single_attempt(
            primary_mask_orig, skeleton_points_orig, skeleton_widths,
            root_id, config_name, category, primary_is_thin,
            mode_name, lateral_mode, seed + attempt,
        )
        if cc8_count == 1:
            result["metadata"]["composite_cc8_retries"] = attempt
            return result

    result["combined_mask"], _ = keep_largest_component(result["combined_mask"])
    result["label_img"][result["combined_mask"] == 0] = 0
    result["metadata"]["composite_cc8_retries"] = MAX_COMPOSITE_RETRIES
    result["metadata"]["composite_cc8_fallback"] = True
    return result


def generate_lateral_root_inline(primary_data, config_name,
                                 lateral_mode_name, seed,
                                 force_taper=None, force_thin=None):
    """Generate laterals for a primary root held in memory.

    This is the recommended entry point for programmatic use. It takes the
    dict returned by :func:`~pyrootsim.roots.primary.generate_single_primary_root`
    and produces the composite (primary + all laterals) mask.

    The function retries up to :data:`MAX_COMPOSITE_RETRIES` times if the
    composite is not 8-connected, falling back to largest-component
    filtering on the last attempt.

    Args:
        primary_data:       dict with ``mask``, ``skeleton_points``,
                            ``skeleton_widths``, and ``metadata``.
        config_name:        primary config key.
        lateral_mode_name:  lateral mode key within the config's category.
        seed:               random seed.
        force_taper:        override taper bucket for all laterals.
        force_thin:         override thin-variant flag for all laterals.

    Returns:
        Dict with ``root_id``, ``combined_mask``, ``label_img``,
        ``primary_mask``, and ``metadata``.
    """
    category = primary_data["metadata"]["category"]
    root_id = primary_data["metadata"]["root_id"]
    primary_is_thin = (
        primary_data["metadata"].get("width_info", {}).get("is_thin", False)
    )

    lateral_mode = LATERAL_MODES[category][lateral_mode_name]

    primary_mask_orig = primary_data["mask"]
    skeleton_points_orig = primary_data["skeleton_points"]
    skeleton_widths = primary_data["skeleton_widths"]

    for attempt in range(MAX_COMPOSITE_RETRIES):
        result, cc8_count = _generate_laterals_single_attempt(
            primary_mask_orig, skeleton_points_orig, skeleton_widths,
            root_id, config_name, category, primary_is_thin,
            lateral_mode_name, lateral_mode, seed + attempt,
            force_taper=force_taper, force_thin=force_thin,
        )
        if cc8_count == 1:
            result["metadata"]["composite_cc8_retries"] = attempt
            return result

    result["combined_mask"], _ = keep_largest_component(result["combined_mask"])
    result["label_img"][result["combined_mask"] == 0] = 0
    result["metadata"]["composite_cc8_retries"] = MAX_COMPOSITE_RETRIES
    result["metadata"]["composite_cc8_fallback"] = True
    return result