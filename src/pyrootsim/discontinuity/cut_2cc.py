"""
Two-connected-component (2cc) occlusion cuts.

Clean two-half-plane cuts with strict cc=2 validation in oriented bounding
boxes (OBBs). Three placement modes:

1. **Standard mid-body** — cuts on primary or lateral root bodies, away from
   tips and junctions.
2. **Lateral-base** — cuts near the junction end of a lateral root without
   touching the primary. Top-tip laterals are heavily prioritised.
3. **Top-tip dedicated** — cuts at the exact junction where a top-tip lateral
   meets the primary tip.

Adaptive sizing: small occlusions use global ranges; large ones scale with
root category and skeleton length.
"""

import numpy as np
import os
import json
from PIL import Image
import cv2
from collections import defaultdict
from datetime import datetime
from skimage.morphology import skeletonize

# ===================================================================
# CONFIG-TO-CATEGORY MAPPING
# ===================================================================

CONFIG_TO_CATEGORY = {
    "01_Short_Kinky_Noisy": "short",
    "02_Short_Smooth_Clean": "short",
    "03_Short_Kinky_Smooth": "short",
    "04_Medium_Kinky_Noisy": "medium",
    "05_Medium_Smooth_Snake": "medium",
    "06_Medium_Clean_GroundTruth": "medium",
    "07_Long_Kinky_Noisy": "long",
    "08_Long_Sweeping_Curves": "long",
    "09_Long_Smooth_Static": "long",
    "10_ExtraLong_Hybrid": "extra_long",
    "11_ExtraLong_Curvy_Clean": "extra_long",
    "12_ExtraLong_Mixed_Sweeping": "extra_long_12",
}

# ===================================================================
# ADAPTIVE SIZE CONFIG
# ===================================================================

GLOBAL_SIZE_BINS = {
    "tiny": (4, 10),
    "small": (8, 18),
}

CATEGORY_SIZE_BINS = {
    "short": {
        "medium": (15, 30),
        "large": (25, 55),
        "max_fraction_of_skeleton": 0.25,
    },
    "medium": {
        "medium": (18, 50),
        "large": (40, 100),
        "max_fraction_of_skeleton": 0.30,
    },
    "long": {
        "medium": (20, 70),
        "large": (55, 160),
        "max_fraction_of_skeleton": 0.35,
    },
    "extra_long": {
        "medium": (25, 100),
        "large": (70, 250),
        "max_fraction_of_skeleton": 0.40,
    },
    "extra_long_12": {
        "medium": (25, 100),
        "large": (70, 250),
        "max_fraction_of_skeleton": 0.40,
    },
}

SIZE_WEIGHTS = {
    "primary": {"tiny": 0.10, "small": 0.20, "medium": 0.40, "large": 0.30},
    "lateral": {"tiny": 0.15, "small": 0.30, "medium": 0.35, "large": 0.20},
    "lateral_base": {"tiny": 0.20, "small": 0.35, "medium": 0.30, "large": 0.15},
}

# ===================================================================
# DENSITY TIERS
# ===================================================================

DENSITY_TIERS = {
    "none": {"prob": 0.05, "count_range": (0, 0)},
    "light": {"prob": 0.20, "count_range": (1, 3)},
    "medium": {"prob": 0.30, "count_range": (4, 8)},
    "heavy": {"prob": 0.30, "count_range": (8, 15)},
    "extreme": {"prob": 0.15, "count_range": (15, 30)},
}

# ===================================================================
# PROTECTION & SPACING PARAMS
# ===================================================================

MIN_SKELETON_DISTANCE_FROM_TIP = 20
MIN_SKELETON_DISTANCE_FROM_JUNCTION = 15
MIN_GAP_BETWEEN_OCCLUSIONS_SKEL = 18
MIN_GLOBAL_DISTANCE = 12
JUNCTION_SAFETY_MARGIN = 25
MAX_ATTEMPTS_PER_ROOT = 10

# Lateral-base cut params
LATERAL_BASE_CUT_ZONE_FRACTION = 0.30
LATERAL_BASE_MIN_LENGTH = 10

# Top-tip detection
TOP_TIP_PRIMARY_FRACTION = 0.06

# Selection bias
TOP_TIP_SELECTION_WEIGHT = 4.0
LATERAL_BASE_PROBABILITY = 0.35

# Top-tip dedicated occlusion
TOP_TIP_OCC_PROBABILITY = 0.25
TOP_TIP_CUT_ZONE_MAX_IDX = 0.15
TOP_TIP_INCLUDE_PRIMARY_TIP = True


# ===================================================================
# LABEL HELPERS
# ===================================================================

def _decode_label(val):
    """Decode a label pixel value into plant ID and root type."""
    if val == 0:
        return None
    plant_id = val // 100
    remainder = val % 100
    if remainder == 0:
        return {"plant_id": plant_id, "root_type": "primary", "lateral_id": None}
    return {"plant_id": plant_id, "root_type": "lateral", "lateral_id": remainder}


# ===================================================================
# SKELETON HELPERS
# ===================================================================

def _get_skeleton_graph(skel_mask):
    """Build adjacency graph from a binary skeleton mask."""
    skel_points = np.argwhere(skel_mask)
    if len(skel_points) < 2:
        return None, None, None
    points_set = set(map(tuple, skel_points))
    adjacency = defaultdict(list)
    for p in points_set:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                n = (p[0] + dy, p[1] + dx)
                if n in points_set:
                    adjacency[p].append(n)
    endpoints = [p for p in points_set if len(adjacency[p]) == 1]
    return points_set, adjacency, endpoints


def _order_skeleton_from_endpoint(start, adjacency, points_set):
    """BFS traversal of skeleton from *start*, returning ordered point array."""
    ordered, visited = [], set()
    queue = [start]
    while queue:
        curr = queue.pop(0)
        if curr in visited:
            continue
        visited.add(curr)
        ordered.append(curr)
        for n in adjacency[curr]:
            if n not in visited:
                queue.append(n)
    return np.array(ordered)


def _compute_geodesic_distances(ordered):
    """Compute cumulative geodesic distances along an ordered skeleton."""
    distances = {tuple(ordered[0]): 0}
    cumulative = 0
    for i in range(1, len(ordered)):
        step = np.sqrt(
            (ordered[i][0] - ordered[i - 1][0]) ** 2
            + (ordered[i][1] - ordered[i - 1][1]) ** 2
        )
        cumulative += step
        distances[tuple(ordered[i])] = cumulative
    return distances


def _identify_primary_endpoints(endpoints):
    """Return (junction, tip) for a primary root — topmost is junction."""
    if len(endpoints) < 2:
        return None, None
    sorted_eps = sorted(endpoints, key=lambda p: p[0])
    return sorted_eps[0], sorted_eps[-1]


def _identify_lateral_endpoints(endpoints, canvas_labels, plant_id):
    """Return (junction, tip) for a lateral root based on proximity to primary."""
    if len(endpoints) < 2:
        return None, None
    primary_tid = plant_id * 100
    h, w = canvas_labels.shape

    def near_primary(pt):
        py, px = pt
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ny, nx = py + dy, px + dx
                if 0 <= ny < h and 0 <= nx < w and canvas_labels[ny, nx] == primary_tid:
                    return True
        return False

    junction, tip = None, None
    for ep in endpoints:
        if near_primary(ep):
            junction = ep
        else:
            tip = ep

    if junction is None or tip is None:
        s = sorted(endpoints, key=lambda p: p[0])
        junction, tip = s[0], s[-1]
    return junction, tip


# ===================================================================
# BUILD ROOT DATA
# ===================================================================

def _find_junction_indices_on_primary(primary_skeleton, lateral_junction_points,
                                      search_radius=8):
    """Map lateral junction points onto the nearest primary skeleton indices."""
    junction_indices = []
    for jpt in lateral_junction_points:
        dists = np.sqrt(
            (primary_skeleton[:, 0] - jpt[0]) ** 2
            + (primary_skeleton[:, 1] - jpt[1]) ** 2
        )
        min_idx = np.argmin(dists)
        if dists[min_idx] < search_radius:
            junction_indices.append(int(min_idx))
    return junction_indices


def _build_roots(canvas_labels, overlap_labels):
    """Build root data structures with top-tip detection.

    This is the standalone (non-vectorised) builder used by
    :func:`process_single_dish_2cc`. For batch pipelines, prefer
    :func:`~pyrootsim.discontinuity.build_roots.build_roots_fast`.
    """
    unique_tids = np.unique(canvas_labels)
    unique_tids = unique_tids[unique_tids > 0]
    h, w = canvas_labels.shape
    roots = {}

    for tid in unique_tids:
        decoded = _decode_label(int(tid))
        if decoded is None:
            continue

        root_pixels = set(map(tuple, np.argwhere(canvas_labels == tid)))
        if overlap_labels is not None:
            for p in np.argwhere(overlap_labels == tid):
                root_pixels.add(tuple(p))
        if len(root_pixels) < 50:
            continue

        root_mask = np.zeros((h, w), dtype=np.uint8)
        for py, px in root_pixels:
            root_mask[py, px] = 1

        skel_mask = skeletonize(root_mask > 0)
        pts_set, adj, endpoints = _get_skeleton_graph(skel_mask)
        if pts_set is None or len(endpoints) < 2:
            continue

        if decoded["root_type"] == "primary":
            junction, tip = _identify_primary_endpoints(endpoints)
        else:
            junction, tip = _identify_lateral_endpoints(
                endpoints, canvas_labels, decoded["plant_id"]
            )

        if junction is None or tip is None:
            continue

        ordered = _order_skeleton_from_endpoint(junction, adj, pts_set)
        geo_junction = _compute_geodesic_distances(ordered)
        ordered_tip = _order_skeleton_from_endpoint(tip, adj, pts_set)
        geo_tip = _compute_geodesic_distances(ordered_tip)

        roots[int(tid)] = {
            "tid": int(tid),
            "plant_id": decoded["plant_id"],
            "root_type": decoded["root_type"],
            "lateral_id": decoded["lateral_id"],
            "root_pixels": root_pixels,
            "root_mask": root_mask,
            "skeleton": ordered,
            "skeleton_set": pts_set,
            "adjacency": adj,
            "tip_end": tip,
            "junction_end": junction,
            "geo_junction": geo_junction,
            "geo_tip": geo_tip,
            "lateral_junction_indices": [],
            "is_top_tip": False,
        }

    # Map lateral junctions onto primaries + detect top-tip
    plants = defaultdict(list)
    for tid, rd in roots.items():
        plants[rd["plant_id"]].append(rd)

    for plant_id, plant_roots in plants.items():
        primary = None
        laterals = []
        for rd in plant_roots:
            if rd["root_type"] == "primary":
                primary = rd
            else:
                laterals.append(rd)
        if primary is None or not laterals:
            continue

        primary_skel_len = len(primary["skeleton"])
        lateral_junctions = [lat["junction_end"] for lat in laterals]
        junction_indices = _find_junction_indices_on_primary(
            primary["skeleton"], lateral_junctions
        )
        primary["lateral_junction_indices"] = junction_indices

        # Top-tip detection: lateral junction within first N% of primary skeleton
        tip_threshold = int(primary_skel_len * TOP_TIP_PRIMARY_FRACTION)
        for lat in laterals:
            jpt = lat["junction_end"]
            dists = np.sqrt(
                (primary["skeleton"][:, 0] - jpt[0]) ** 2
                + (primary["skeleton"][:, 1] - jpt[1]) ** 2
            )
            min_idx = np.argmin(dists)
            if dists[min_idx] < 10 and min_idx <= tip_threshold:
                lat["is_top_tip"] = True

    return roots


# ===================================================================
# GLOBAL OCCLUSION TRACKER
# ===================================================================

class GlobalOcclusionTracker:
    """Tracks placed occlusion pixels for minimum-distance and edge-margin enforcement."""

    def __init__(self, canvas_shape, min_distance=12, canvas_edge_margin=18):
        self.h, self.w = canvas_shape
        self.min_distance = min_distance
        self.canvas_edge_margin = canvas_edge_margin
        self.proximity_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * min_distance + 1, 2 * min_distance + 1),
        )
        self.occupied_per_root = defaultdict(list)

    def check(self, erase_pixels):
        """Return True if *erase_pixels* are far enough from prior occlusions and canvas edges."""
        margin = self.canvas_edge_margin
        for py, px in erase_pixels:
            if py < margin or py >= self.h - margin:
                return False
            if px < margin or px >= self.w - margin:
                return False
            if 0 <= py < self.h and 0 <= px < self.w:
                if self.proximity_mask[py, px] > 0:
                    return False
        return True

    def commit(self, erase_pixels, tid=None, skel_range=None):
        """Register a placed occlusion, dilating the proximity mask."""
        erase_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        for py, px in erase_pixels:
            if 0 <= py < self.h and 0 <= px < self.w:
                erase_mask[py, px] = 255
        dilated = cv2.dilate(erase_mask, self.kernel, iterations=1)
        self.proximity_mask = np.maximum(self.proximity_mask, dilated)
        if tid is not None and skel_range is not None:
            self.occupied_per_root[tid].append(skel_range)

    def get_skel_occupied(self, tid):
        """Return list of occupied skeleton-index ranges for a given root TID."""
        return self.occupied_per_root.get(tid, [])


# ===================================================================
# CUT SIZE SAMPLING
# ===================================================================

def _get_size_bins(category):
    """Get full size-bin ranges for a category."""
    cat_bins = CATEGORY_SIZE_BINS.get(category, CATEGORY_SIZE_BINS["medium"])
    return {
        "tiny": GLOBAL_SIZE_BINS["tiny"],
        "small": GLOBAL_SIZE_BINS["small"],
        "medium": cat_bins["medium"],
        "large": cat_bins["large"],
    }


def _sample_cut_length(root_data, category, cut_type, rng):
    """Sample a cut length and size bin.

    Returns
    -------
    tuple of (int, str)
        ``(cut_length, size_bin)``
    """
    bins = _get_size_bins(category)
    weights = SIZE_WEIGHTS[cut_type]
    bin_names = list(weights.keys())
    probs = np.array([weights[b] for b in bin_names])
    probs /= probs.sum()

    size_bin = rng.choice(bin_names, p=probs)
    bin_lo, bin_hi = bins[size_bin]

    skel_len = len(root_data["skeleton"])
    max_frac = CATEGORY_SIZE_BINS.get(category, CATEGORY_SIZE_BINS["medium"])[
        "max_fraction_of_skeleton"
    ]
    abs_max = max(10, int(skel_len * max_frac))

    actual_lo = max(4, bin_lo)
    actual_hi = min(abs_max, bin_hi)
    if actual_lo >= actual_hi:
        actual_hi = actual_lo + 1

    return int(rng.integers(actual_lo, actual_hi + 1)), size_bin


# ===================================================================
# DENSITY SAMPLING
# ===================================================================

def _sample_density_tier(rng):
    """Sample a density tier name."""
    tiers = list(DENSITY_TIERS.keys())
    probs = np.array([DENSITY_TIERS[t]["prob"] for t in tiers])
    probs /= probs.sum()
    return rng.choice(tiers, p=probs)


def _sample_target_count(tier, num_roots, rng):
    """Sample an occlusion target count for a given tier and root count."""
    lo, hi = DENSITY_TIERS[tier]["count_range"]
    if hi == 0:
        return 0
    hi = min(hi, num_roots * 2)
    lo = min(lo, hi)
    return int(rng.integers(lo, hi + 1))


# ===================================================================
# CLEAN CUT (two half-planes)
# ===================================================================

def _get_local_direction(skeleton, idx, window=7):
    """Estimate local skeleton direction at *idx* using a ±window average."""
    start = max(0, idx - window)
    end = min(len(skeleton), idx + window + 1)
    if end - start < 2:
        return np.array([1, 0])
    pts = skeleton[start:end]
    d = pts[-1].astype(float) - pts[0].astype(float)
    norm = np.linalg.norm(d)
    return d / norm if norm > 1e-6 else np.array([1, 0])


def _compute_clean_cut(root_data, center_idx, cut_length, canvas_labels,
                       restrict_to_tid=None):
    """Compute a clean two-half-plane cut on a root.

    Parameters
    ----------
    root_data : dict
        Root data dictionary.
    center_idx : int
        Skeleton index of the cut centre.
    cut_length : int
        Total cut length in skeleton indices.
    canvas_labels : numpy.ndarray
        Label image.
    restrict_to_tid : int or None
        If set, only erase pixels with this exact label (used for
        lateral-base cuts to avoid touching the primary).

    Returns
    -------
    tuple of (set, dict or None)
        ``(erase_pixels, cut_geometry)`` or ``(set(), None)`` on failure.
    """
    skeleton = root_data["skeleton"]
    tid = root_data["tid"] if restrict_to_tid is None else restrict_to_tid
    root_mask = root_data["root_mask"]
    h, w = canvas_labels.shape

    half = cut_length // 2
    start_idx = max(0, center_idx - half)
    end_idx = min(len(skeleton) - 1, center_idx + half)
    if end_idx <= start_idx:
        return set(), None

    dir_start = _get_local_direction(skeleton, start_idx)
    pt_start = skeleton[start_idx].astype(float)
    dir_end = _get_local_direction(skeleton, end_idx)
    pt_end = skeleton[end_idx].astype(float)
    center_pt = skeleton[center_idx].astype(float)

    normal_start = dir_start.copy()
    if np.dot(normal_start, center_pt - pt_start) < 0:
        normal_start = -normal_start
    normal_end = dir_end.copy()
    if np.dot(normal_end, center_pt - pt_end) < 0:
        normal_end = -normal_end

    cut_skel = skeleton[start_idx : end_idx + 1]
    margin = 20
    min_y = max(0, int(cut_skel[:, 0].min()) - margin)
    max_y = min(h - 1, int(cut_skel[:, 0].max()) + margin)
    min_x = max(0, int(cut_skel[:, 1].min()) - margin)
    max_x = min(w - 1, int(cut_skel[:, 1].max()) + margin)

    roi_root = root_mask[min_y : max_y + 1, min_x : max_x + 1] > 0
    roi_labels = canvas_labels[min_y : max_y + 1, min_x : max_x + 1] == tid
    roi_candidate = roi_root & roi_labels
    ys_local, xs_local = np.where(roi_candidate)
    if len(ys_local) == 0:
        return set(), None

    ys_global = ys_local + min_y
    xs_global = xs_local + min_x
    points = np.column_stack([ys_global.astype(float), xs_global.astype(float)])

    dot_start = (
        (points[:, 0] - pt_start[0]) * normal_start[0]
        + (points[:, 1] - pt_start[1]) * normal_start[1]
    )
    dot_end = (
        (points[:, 0] - pt_end[0]) * normal_end[0]
        + (points[:, 1] - pt_end[1]) * normal_end[1]
    )

    between = (dot_start >= 0) & (dot_end >= 0)
    erase_pixels = set(
        (int(ys_global[i]), int(xs_global[i])) for i in np.where(between)[0]
    )

    cut_geometry = {
        "pt_start": pt_start,
        "pt_end": pt_end,
        "dir_start": dir_start,
        "dir_end": dir_end,
        "start_idx": start_idx,
        "end_idx": end_idx,
    }
    return erase_pixels, cut_geometry


# ===================================================================
# OBB
# ===================================================================

def _compute_obb_2cc(erase_pixels, cut_geometry, root_mask, trial_mask):
    """Compute a surgically tight OBB that grabs the two severed ends."""
    if not erase_pixels:
        return None, None

    ec = np.array(list(erase_pixels))
    ey1, ex1 = ec.min(axis=0)
    ey2, ex2 = ec.max(axis=0)
    h, w = trial_mask.shape

    pad = 15
    y1, y2 = max(0, ey1 - pad), min(h, ey2 + pad + 1)
    x1, x2 = max(0, ex1 - pad), min(w, ex2 + pad + 1)

    local_erase = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    for py, px in erase_pixels:
        local_erase[py - y1, px - x1] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    grab_zone = cv2.dilate(local_erase, kernel, iterations=1)
    local_occ = trial_mask[y1:y2, x1:x2]
    last_parts = (grab_zone > 0) & (local_occ > 0)

    sy_local, sx_local = np.where(last_parts)
    syg, sxg = sy_local + y1, sx_local + x1

    ay = np.concatenate([ec[:, 0], syg])
    ax = np.concatenate([ec[:, 1], sxg])

    if len(ay) < 3:
        return None, None

    cp = np.column_stack([ax, ay]).astype(np.float32)
    r = cv2.minAreaRect(cp)
    b = cv2.boxPoints(r)

    bn = b.copy()
    bn[:, 0] /= w
    bn[:, 1] /= h
    return b, bn


def _count_cc_in_obb(occluded_mask, obb_box):
    """Count connected components inside an OBB region of the occluded mask."""
    if obb_box is None:
        return -1
    h, w = occluded_mask.shape
    obb_int = obb_box.astype(np.int32).reshape((-1, 1, 2))
    obb_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(obb_mask, [obb_int], 255)
    inside = (obb_mask > 0) & (occluded_mask > 0)
    nl, _ = cv2.connectedComponents(inside.astype(np.uint8), connectivity=8)
    return nl - 1


# ===================================================================
# VALIDATION
# ===================================================================

def _validate_2cc(erase_pixels, current_mask, trial_mask, padding=40,
                  min_stub_size=5):
    """Validate a 2cc cut using a localised padded patch comparison.

    Compares the dish before the cut vs after the cut within a local
    window. The cut must increase the connected-component count by
    exactly 1.
    """
    if not erase_pixels:
        return False

    ec = np.array(list(erase_pixels))
    ey1, ex1 = ec.min(axis=0)
    ey2, ex2 = ec.max(axis=0)

    h, w = current_mask.shape

    y1, y2 = max(0, ey1 - padding), min(h, ey2 + padding + 1)
    x1, x2 = max(0, ex1 - padding), min(w, ex2 + padding + 1)

    patch_before = (current_mask[y1:y2, x1:x2] > 0).astype(np.uint8) * 255
    nl_before, _ = cv2.connectedComponents(patch_before, connectivity=8)

    patch_after = (trial_mask[y1:y2, x1:x2] > 0).astype(np.uint8) * 255
    nl_after, labels, stats, _ = cv2.connectedComponentsWithStats(
        patch_after, connectivity=8
    )

    if (nl_after - nl_before) != 1:
        return False

    for i in range(1, nl_after):
        if stats[i, cv2.CC_STAT_AREA] < min_stub_size:
            return False

    return True


def _check_junction_safety(erase_pixels, canvas_labels, plant_id, tid):
    """Ensure we don't erase pixels belonging to another root of the same plant."""
    for py, px in erase_pixels:
        lbl = canvas_labels[py, px]
        if lbl != 0 and lbl != tid:
            d = _decode_label(int(lbl))
            if d and d["plant_id"] == plant_id:
                return False
    return True


def _check_overlap_safety(erase_pixels, overlap_labels):
    """Return False if any erase pixel falls in the overlap region."""
    if overlap_labels is None:
        return True
    for py, px in erase_pixels:
        if overlap_labels[py, px] > 0:
            return False
    return True


def _check_disconnects_lateral(root_data, erase_pixels, roots):
    """For primary cuts, ensure we don't disconnect a lateral from its junction."""
    if root_data["root_type"] != "primary":
        return True
    plant_id = root_data["plant_id"]
    for tid, rd in roots.items():
        if rd["plant_id"] != plant_id or rd["root_type"] != "lateral":
            continue
        jpt = rd["junction_end"]
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                ny, nx = jpt[0] + dy, jpt[1] + dx
                if (ny, nx) in erase_pixels:
                    return False
    return True


# ===================================================================
# FIND VALID POSITIONS: standard (primary / lateral mid-body)
# ===================================================================

def _find_valid_positions_standard(root_data, cut_length, tracker):
    """Find valid cut positions on a root, avoiding tips, junctions, and occupied zones."""
    skeleton = root_data["skeleton"]
    geo_tip = root_data["geo_tip"]
    geo_junction = root_data["geo_junction"]
    skel_len = len(skeleton)
    junction_indices = root_data.get("lateral_junction_indices", [])
    tid = root_data["tid"]

    tip_prot = max(MIN_SKELETON_DISTANCE_FROM_TIP, int(skel_len * 0.05))
    junc_prot = max(MIN_SKELETON_DISTANCE_FROM_JUNCTION, int(skel_len * 0.03))
    half = cut_length // 2

    forbidden = set()
    for ji in junction_indices:
        for offset in range(-JUNCTION_SAFETY_MARGIN, JUNCTION_SAFETY_MARGIN + 1):
            idx = ji + offset
            if 0 <= idx < skel_len:
                forbidden.add(idx)

    occupied = tracker.get_skel_occupied(tid)
    candidates = []
    for idx in range(half, skel_len - half):
        pt = tuple(skeleton[idx])
        if geo_tip.get(pt, 0) < tip_prot:
            continue
        if geo_junction.get(pt, 0) < junc_prot:
            continue
        if idx in forbidden:
            continue
        cs, ce = idx - half, idx + half
        if any(
            not (
                ce + MIN_GAP_BETWEEN_OCCLUSIONS_SKEL < os_
                or cs - MIN_GAP_BETWEEN_OCCLUSIONS_SKEL > oe
            )
            for os_, oe in occupied
        ):
            continue
        candidates.append(idx)
    return candidates


# ===================================================================
# FIND VALID POSITIONS: lateral-base
# ===================================================================

def _find_valid_positions_lateral_base(lateral_data, cut_length, tracker):
    """Find valid cut positions near the junction (base) of a lateral root."""
    skeleton = lateral_data["skeleton"]
    skel_len = len(skeleton)
    tid = lateral_data["tid"]
    half = cut_length // 2

    if skel_len < LATERAL_BASE_MIN_LENGTH:
        return []

    max_idx = int(skel_len * LATERAL_BASE_CUT_ZONE_FRACTION)
    max_idx = max(half + 1, max_idx)

    occupied = tracker.get_skel_occupied(tid)
    candidates = []

    for idx in range(half, min(max_idx, skel_len - half)):
        cs, ce = idx - half, idx + half
        if any(
            not (
                ce + MIN_GAP_BETWEEN_OCCLUSIONS_SKEL < os_
                or cs - MIN_GAP_BETWEEN_OCCLUSIONS_SKEL > oe
            )
            for os_, oe in occupied
        ):
            continue
        candidates.append(idx)
    return candidates


# ===================================================================
# APPLY OCCLUSION
# ===================================================================

def _apply_occlusion(canvas_mask, erase_pixels, tid, canvas_labels):
    """Erase pixels belonging to *tid* from the canvas mask."""
    result = canvas_mask.copy()
    h, w = result.shape
    for py, px in erase_pixels:
        if 0 <= py < h and 0 <= px < w:
            if canvas_labels[py, px] == tid:
                result[py, px] = 0
    return result


# ===================================================================
# TRY PLACE: STANDARD 2CC (primary or lateral mid-body)
# ===================================================================

def _try_place_standard_2cc(root_data, roots, canvas_labels, overlap_labels,
                            current_mask, tracker, category, rng):
    """Attempt one standard 2cc occlusion on a root's mid-body."""
    tid = root_data["tid"]
    cut_type = root_data["root_type"]
    cut_length, size_bin = _sample_cut_length(root_data, category, cut_type, rng)

    positions = _find_valid_positions_standard(root_data, cut_length, tracker)

    if not positions:
        for frac in [0.5, 0.3]:
            smaller = max(4, int(cut_length * frac))
            positions = _find_valid_positions_standard(root_data, smaller, tracker)
            if positions:
                cut_length = smaller
                size_bin = "tiny" if smaller <= 10 else "small"
                break
        if not positions:
            return None, current_mask

    rng.shuffle(positions)

    for center_idx in positions[:15]:
        erase, cut_geo = _compute_clean_cut(
            root_data, center_idx, cut_length, canvas_labels
        )
        if len(erase) < 3:
            continue
        if not _check_junction_safety(erase, canvas_labels, root_data["plant_id"], tid):
            continue
        if not _check_overlap_safety(erase, overlap_labels):
            continue
        if cut_type == "primary":
            if not _check_disconnects_lateral(root_data, erase, roots):
                continue
        if not tracker.check(erase):
            continue

        trial = _apply_occlusion(current_mask, erase, tid, canvas_labels)

        if not _validate_2cc(erase, current_mask, trial):
            continue

        obb_box, obb_norm = _compute_obb_2cc(
            erase, cut_geo, root_data["root_mask"], trial
        )
        cc = _count_cc_in_obb(trial, obb_box)

        half = cut_length // 2
        tracker.commit(
            erase,
            tid=tid,
            skel_range=(
                center_idx - half - MIN_GAP_BETWEEN_OCCLUSIONS_SKEL,
                center_idx + half + MIN_GAP_BETWEEN_OCCLUSIONS_SKEL,
            ),
        )

        return {
            "tid": tid,
            "plant_id": root_data["plant_id"],
            "root_type": cut_type,
            "center_idx": int(center_idx),
            "cut_length": int(cut_length),
            "size_bin": size_bin,
            "erase_pixels": erase,
            "cut_geometry": cut_geo,
            "obb_box": obb_box,
            "obb_norm": obb_norm,
            "cc_in_obb": cc,
            "occlusion_type": "2cc",
            "is_lateral_base": False,
            "is_top_tip_target": False,
        }, trial

    return None, current_mask


# ===================================================================
# TRY PLACE: LATERAL-BASE 2CC
# ===================================================================

def _find_top_tip_laterals(roots):
    """Return list of lateral root dicts flagged as top-tip."""
    return [
        rd
        for rd in roots.values()
        if rd["root_type"] == "lateral" and rd.get("is_top_tip", False)
    ]


def _find_primary_for_lateral(lateral_data, roots):
    """Find the primary root that this lateral belongs to."""
    pid = lateral_data["plant_id"]
    for rd in roots.values():
        if rd["plant_id"] == pid and rd["root_type"] == "primary":
            return rd
    return None


def _try_place_lateral_base_2cc(lateral_data, roots, canvas_labels, overlap_labels,
                                current_mask, tracker, category, rng):
    """Attempt a 2cc occlusion near the base (junction end) of a lateral root.

    Only erases pixels with the lateral's own TID — never touches the primary.
    """
    tid = lateral_data["tid"]
    cut_length, size_bin = _sample_cut_length(
        lateral_data, category, "lateral_base", rng
    )

    positions = _find_valid_positions_lateral_base(lateral_data, cut_length, tracker)

    if not positions:
        for frac in [0.5, 0.3]:
            smaller = max(4, int(cut_length * frac))
            positions = _find_valid_positions_lateral_base(
                lateral_data, smaller, tracker
            )
            if positions:
                cut_length = smaller
                size_bin = "tiny" if smaller <= 10 else "small"
                break
        if not positions:
            return None, current_mask

    rng.shuffle(positions)

    primary_rd = _find_primary_for_lateral(lateral_data, roots)
    combined_mask = lateral_data["root_mask"].copy()
    if primary_rd is not None:
        combined_mask = np.maximum(combined_mask, primary_rd["root_mask"])

    for center_idx in positions[:10]:
        erase, cut_geo = _compute_clean_cut(
            lateral_data, center_idx, cut_length, canvas_labels, restrict_to_tid=tid
        )
        if len(erase) < 3:
            continue
        if not _check_overlap_safety(erase, overlap_labels):
            continue
        if not tracker.check(erase):
            continue

        trial = _apply_occlusion(current_mask, erase, tid, canvas_labels)

        if not _validate_2cc(erase, current_mask, trial):
            continue

        obb_box, obb_norm = _compute_obb_2cc(erase, cut_geo, combined_mask, trial)
        cc = _count_cc_in_obb(trial, obb_box)

        half = cut_length // 2
        tracker.commit(
            erase,
            tid=tid,
            skel_range=(
                center_idx - half - MIN_GAP_BETWEEN_OCCLUSIONS_SKEL,
                center_idx + half + MIN_GAP_BETWEEN_OCCLUSIONS_SKEL,
            ),
        )

        return {
            "tid": tid,
            "plant_id": lateral_data["plant_id"],
            "root_type": "lateral",
            "center_idx": int(center_idx),
            "cut_length": int(cut_length),
            "size_bin": size_bin,
            "erase_pixels": erase,
            "cut_geometry": cut_geo,
            "obb_box": obb_box,
            "obb_norm": obb_norm,
            "cc_in_obb": cc,
            "occlusion_type": "2cc",
            "is_lateral_base": True,
            "is_top_tip_target": lateral_data.get("is_top_tip", False),
        }, trial

    return None, current_mask


# ===================================================================
# TRY PLACE: TOP-TIP DEDICATED 2CC
# ===================================================================

def _try_place_top_tip_2cc(lateral_data, roots, canvas_labels, overlap_labels,
                           current_mask, tracker, category, rng):
    """Attempt a 2cc occlusion at a top-tip lateral junction."""
    lat_tid = lateral_data["tid"]
    lat_skel = lateral_data["skeleton"]
    lat_skel_len = len(lat_skel)

    if lat_skel_len < LATERAL_BASE_MIN_LENGTH:
        return None, current_mask

    cut_length, size_bin = _sample_cut_length(
        lateral_data, category, "lateral_base", rng
    )
    half = cut_length // 2

    max_zone = max(half + 1, int(lat_skel_len * TOP_TIP_CUT_ZONE_MAX_IDX))
    max_zone = min(max_zone, lat_skel_len - half - 1)

    if max_zone <= half:
        cut_length = max(4, min(cut_length, lat_skel_len // 3))
        half = cut_length // 2
        max_zone = max(half + 1, int(lat_skel_len * TOP_TIP_CUT_ZONE_MAX_IDX))
        max_zone = min(max_zone, lat_skel_len - half - 1)
        if max_zone <= half:
            return None, current_mask
        size_bin = "tiny" if cut_length <= 10 else "small"

    occupied = tracker.get_skel_occupied(lat_tid)

    candidates = []
    for idx in range(half, max_zone + 1):
        cs, ce = idx - half, idx + half
        if any(
            not (
                ce + MIN_GAP_BETWEEN_OCCLUSIONS_SKEL < os_
                or cs - MIN_GAP_BETWEEN_OCCLUSIONS_SKEL > oe
            )
            for os_, oe in occupied
        ):
            continue
        candidates.append(idx)

    if not candidates:
        return None, current_mask

    rng.shuffle(candidates)
    primary_rd = _find_primary_for_lateral(lateral_data, roots)

    if primary_rd is None:
        return None, current_mask

    for center_idx in candidates[:8]:
        erase_lat, cut_geo = _compute_clean_cut(
            lateral_data, center_idx, cut_length, canvas_labels,
            restrict_to_tid=lat_tid,
        )
        if len(erase_lat) < 3:
            continue

        all_erase = set(erase_lat)

        if TOP_TIP_INCLUDE_PRIMARY_TIP and primary_rd is not None:
            pri_skel = primary_rd["skeleton"]
            pri_tid = primary_rd["tid"]
            pri_tip_cut = max(3, cut_length // 4)
            pri_center = min(pri_tip_cut // 2 + 1, len(pri_skel) - 1)
            erase_pri, _ = _compute_clean_cut(
                primary_rd, pri_center, pri_tip_cut, canvas_labels,
                restrict_to_tid=pri_tid,
            )
            if len(erase_pri) > 0 and len(erase_pri) < len(erase_lat) * 2:
                all_erase |= erase_pri

        if not _check_overlap_safety(all_erase, overlap_labels):
            continue
        if not tracker.check(all_erase):
            continue

        trial = current_mask.copy()
        h, w = trial.shape
        for py, px in all_erase:
            lbl = canvas_labels[py, px]
            if lbl == lat_tid or (primary_rd is not None and lbl == primary_rd["tid"]):
                trial[py, px] = 0

        obb_box, obb_norm = _compute_obb_2cc(
            erase_lat, cut_geo, lateral_data["root_mask"], trial
        )
        cc = _count_cc_in_obb(trial, obb_box)

        tracker.commit(
            all_erase,
            tid=lat_tid,
            skel_range=(
                center_idx - half - MIN_GAP_BETWEEN_OCCLUSIONS_SKEL,
                center_idx + half + MIN_GAP_BETWEEN_OCCLUSIONS_SKEL,
            ),
        )

        return {
            "tid": lat_tid,
            "plant_id": lateral_data["plant_id"],
            "root_type": "lateral",
            "center_idx": int(center_idx),
            "cut_length": int(cut_length),
            "size_bin": size_bin,
            "erase_pixels": all_erase,
            "cut_geometry": cut_geo,
            "obb_box": obb_box,
            "obb_norm": obb_norm,
            "cc_in_obb": cc,
            "occlusion_type": "2cc_top_tip",
            "is_lateral_base": True,
            "is_top_tip_target": True,
            "primary_tip_erased": TOP_TIP_INCLUDE_PRIMARY_TIP and primary_rd is not None,
        }, trial

    return None, current_mask


# ===================================================================
# CORE: PLACE ALL 2CC ON A DISH
# ===================================================================

def place_2cc_occlusions(roots, canvas_labels, overlap_labels, category, rng,
                         target_count, tracker=None, current_mask=None):
    """Place 2cc occlusions on a dish with three placement modes.

    Parameters
    ----------
    roots : dict
        Root data structures (from ``build_roots_fast`` or ``_build_roots``).
    canvas_labels : numpy.ndarray
        Label image.
    overlap_labels : numpy.ndarray or None
        Overlap label image.
    category : str
        Root category (e.g. ``"short"``, ``"medium"``).
    rng : numpy.random.Generator
        Random number generator.
    target_count : int
        Target number of occlusions to place.
    tracker : GlobalOcclusionTracker or None
        Optional pre-existing tracker (created internally if None).
    current_mask : numpy.ndarray or None
        Optional pre-occluded mask (created from canvas_labels if None).

    Returns
    -------
    tuple of (list, numpy.ndarray)
        ``(all_occlusions, occluded_mask)``
    """
    if current_mask is None:
        current_mask = (canvas_labels > 0).astype(np.uint8) * 255
    else:
        current_mask = current_mask.copy()

    all_occlusions = []

    if target_count == 0:
        return all_occlusions, current_mask

    h, w = canvas_labels.shape
    if tracker is None:
        tracker = GlobalOcclusionTracker((h, w))

    all_roots = list(roots.values())
    primaries = [r for r in all_roots if r["root_type"] == "primary"]
    laterals = [r for r in all_roots if r["root_type"] == "lateral"]
    top_tip_laterals = _find_top_tip_laterals(roots)

    if not all_roots:
        return all_occlusions, current_mask

    stalled = 0
    max_stall = len(all_roots) * 5

    while len(all_occlusions) < target_count and stalled < max_stall:
        stalled += 1
        occ = None

        roll = rng.random()

        # Mode 1: Top-tip dedicated
        if roll < TOP_TIP_OCC_PROBABILITY and top_tip_laterals:
            lat_rd = top_tip_laterals[rng.integers(0, len(top_tip_laterals))]
            occ, current_mask = _try_place_top_tip_2cc(
                lat_rd, roots, canvas_labels, overlap_labels,
                current_mask, tracker, category, rng,
            )

        # Mode 2: Lateral-base (general)
        elif roll < TOP_TIP_OCC_PROBABILITY + LATERAL_BASE_PROBABILITY and laterals:
            weights = []
            for lat in laterals:
                w_val = TOP_TIP_SELECTION_WEIGHT if lat.get("is_top_tip", False) else 1.0
                weights.append(w_val)
            weights = np.array(weights)
            weights /= weights.sum()
            lat_rd = laterals[rng.choice(len(laterals), p=weights)]

            occ, current_mask = _try_place_lateral_base_2cc(
                lat_rd, roots, canvas_labels, overlap_labels,
                current_mask, tracker, category, rng,
            )

        # Mode 3: Standard mid-body
        else:
            rd = all_roots[rng.integers(0, len(all_roots))]
            occ, current_mask = _try_place_standard_2cc(
                rd, roots, canvas_labels, overlap_labels,
                current_mask, tracker, category, rng,
            )

        if occ is not None:
            all_occlusions.append(occ)
            stalled = 0

    return all_occlusions, current_mask


# ===================================================================
# PROCESS SINGLE DISH (entry point)
# ===================================================================

def process_single_dish_2cc(mask_path, config_name, output_dir, seed=None):
    """Process one petri dish for 2cc occlusions.

    Standalone entry point that loads images, builds roots, samples a
    density tier, places occlusions, and writes outputs to disk.

    Parameters
    ----------
    mask_path : str
        Path to the ``*_mask.png`` file.
    config_name : str
        Root configuration name (key into ``CONFIG_TO_CATEGORY``).
    output_dir : str
        Directory for output files.
    seed : int or None
        RNG seed (derived from dish ID if None).

    Returns
    -------
    dict
        Statistics dictionary with keys like ``dish_id``, ``success``,
        ``placed``, etc.
    """
    try:
        dish_id = os.path.basename(mask_path).replace("_mask.png", "")
        labels_path = mask_path.replace("_mask.png", "_labels.png")
        overlap_path = mask_path.replace("_mask.png", "_overlap.png")
        meta_path = mask_path.replace("_mask.png", "_metadata.json")

        if not os.path.exists(labels_path) or not os.path.exists(meta_path):
            return {"dish_id": dish_id, "success": False, "error": "missing_files"}

        canvas_mask = np.array(Image.open(mask_path))
        canvas_labels = np.array(Image.open(labels_path))
        overlap_labels = (
            np.array(Image.open(overlap_path))
            if os.path.exists(overlap_path)
            else None
        )

        category = CONFIG_TO_CATEGORY.get(config_name, "medium")

        if seed is None:
            seed = hash(dish_id) % (2 ** 31)
        rng = np.random.default_rng(seed)

        roots = _build_roots(canvas_labels, overlap_labels)

        tier = _sample_density_tier(rng)
        target = _sample_target_count(tier, len(roots), rng)

        occlusions, occluded_mask = place_2cc_occlusions(
            roots, canvas_labels, overlap_labels, category, rng, target
        )

        # Save outputs
        os.makedirs(output_dir, exist_ok=True)

        Image.fromarray(occluded_mask).save(
            os.path.join(output_dir, f"{dish_id}_mask_occluded.png")
        )

        obb_lines = []
        for occ in occlusions:
            if occ["obb_norm"] is not None:
                bn = occ["obb_norm"]
                line = "0 " + " ".join(
                    f"{bn[i, 0]:.6f} {bn[i, 1]:.6f}" for i in range(4)
                )
                obb_lines.append(line)

        with open(os.path.join(output_dir, f"{dish_id}_obb.txt"), "w") as f:
            for line in obb_lines:
                f.write(line + "\n")

        occ_meta = {
            "dish_id": dish_id,
            "config": config_name,
            "category": category,
            "density_tier": tier,
            "target_count": target,
            "placed_count": len(occlusions),
            "num_roots": len(roots),
            "occlusions": [
                {
                    "tid": int(o["tid"]),
                    "plant_id": int(o["plant_id"]),
                    "root_type": o["root_type"],
                    "cut_length": o["cut_length"],
                    "size_bin": o["size_bin"],
                    "cc_in_obb": o["cc_in_obb"],
                    "is_lateral_base": o["is_lateral_base"],
                    "is_top_tip_target": o["is_top_tip_target"],
                    "obb_xy": (
                        [[float(p[0]), float(p[1])] for p in o["obb_box"]]
                        if o["obb_box"] is not None
                        else None
                    ),
                }
                for o in occlusions
            ],
            "generation_timestamp": datetime.now().isoformat(),
            "seed": int(seed),
        }

        with open(
            os.path.join(output_dir, f"{dish_id}_occlusion_meta.json"), "w"
        ) as f:
            json.dump(occ_meta, f, indent=2)

        return {
            "dish_id": dish_id,
            "success": True,
            "tier": tier,
            "target": target,
            "placed": len(occlusions),
            "num_roots": len(roots),
            "lateral_base_count": sum(
                1 for o in occlusions if o["is_lateral_base"]
            ),
            "top_tip_count": sum(
                1 for o in occlusions if o["is_top_tip_target"]
            ),
            "size_distribution": (
                dict(
                    zip(
                        *np.unique(
                            [o["size_bin"] for o in occlusions], return_counts=True
                        )
                    )
                )
                if occlusions
                else {}
            ),
        }

    except Exception as e:
        dish_id = os.path.basename(mask_path).replace("_mask.png", "")
        return {"dish_id": dish_id, "success": False, "error": str(e)}