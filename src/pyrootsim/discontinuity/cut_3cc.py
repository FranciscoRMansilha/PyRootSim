"""
Three-connected-component (3cc) occlusion cuts.

Junction-targeting cuts on primary roots with optional lateral extension,
producing strict cc=3 in oriented bounding boxes (OBBs). Two placement modes:

1. **Standard junction** — cuts centred on isolated primary–lateral junctions,
   with optional extension into the lateral body.
2. **Top-tip junction** — dedicated placement at top-tip lateral junctions
   where a single lateral meets the primary tip.

Adaptive sizing with a deferred big-occlusion queue to maximise placement
success.
"""

import numpy as np
import os
import glob
import json
import random
from PIL import Image
import cv2
from collections import defaultdict
from datetime import datetime
from skimage.morphology import skeletonize

# ===================================================================
# SIZE CONFIG PER CATEGORY
# ===================================================================

SIZE_CONFIG_3CC = {
    "short": {
        "primary_cut_range": (15, 40),
        "lateral_ext_range": (5, 20),
        "max_primary_fraction": 0.20,
        "max_lateral_fraction": 0.25,
        "big_primary_range": (30, 40),
        "big_lateral_range": (15, 20),
    },
    "medium": {
        "primary_cut_range": (18, 70),
        "lateral_ext_range": (8, 35),
        "max_primary_fraction": 0.25,
        "max_lateral_fraction": 0.30,
        "big_primary_range": (50, 70),
        "big_lateral_range": (25, 35),
    },
    "long": {
        "primary_cut_range": (20, 100),
        "lateral_ext_range": (10, 50),
        "max_primary_fraction": 0.30,
        "max_lateral_fraction": 0.35,
        "big_primary_range": (70, 100),
        "big_lateral_range": (35, 50),
    },
    "extra_long": {
        "primary_cut_range": (20, 140),
        "lateral_ext_range": (10, 60),
        "max_primary_fraction": 0.35,
        "max_lateral_fraction": 0.40,
        "big_primary_range": (90, 140),
        "big_lateral_range": (40, 60),
    },
    "extra_long_12": {
        "primary_cut_range": (20, 140),
        "lateral_ext_range": (10, 65),
        "max_primary_fraction": 0.35,
        "max_lateral_fraction": 0.40,
        "big_primary_range": (90, 140),
        "big_lateral_range": (45, 65),
    },
}

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
# PROTECTION PARAMS
# ===================================================================

MIN_SKELETON_DISTANCE_FROM_TIP = 20
MIN_SKELETON_DISTANCE_FROM_JUNCTION = 15
MIN_GAP_BETWEEN_OCCLUSIONS = 18
MIN_GLOBAL_DISTANCE = 12
JUNCTION_TARGET_TOLERANCE = 10
MIN_JUNCTION_SEPARATION = 40
PROB_LATERAL_EXTENSION = 0.5
FRAGMENT_PROXIMITY_RADIUS = 5
MIN_FRAGMENT_SIZE = 15

BIG_OCCLUSION_MAX_DEFER = 10

# Top-tip 3cc
TOP_TIP_PRIMARY_FRACTION = 0.06
TOP_TIP_3CC_PROBABILITY = 0.30
TOP_TIP_CUT_ZONE_BELOW_JUNCTION = 15


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
        """Return True if *erase_pixels* pass proximity and edge checks."""
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
        """Register a placed occlusion."""
        erase_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        for py, px in erase_pixels:
            if 0 <= py < self.h and 0 <= px < self.w:
                erase_mask[py, px] = 255
        dilated = cv2.dilate(erase_mask, self.kernel, iterations=1)
        self.proximity_mask = np.maximum(self.proximity_mask, dilated)
        if tid is not None and skel_range is not None:
            self.occupied_per_root[tid].append(skel_range)

    def get_skel_occupied(self, tid):
        """Return occupied skeleton-index ranges for a root TID."""
        return self.occupied_per_root.get(tid, [])


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
    """BFS traversal of skeleton from *start*."""
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
    """Return (junction, tip) for a primary root."""
    if len(endpoints) < 2:
        return None, None
    sorted_eps = sorted(endpoints, key=lambda p: p[0])
    return sorted_eps[0], sorted_eps[-1]


def _identify_lateral_endpoints(endpoints, canvas_labels, plant_id):
    """Return (junction, tip) for a lateral root."""
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
    """Map lateral junction points onto nearest primary skeleton indices."""
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
    """Build root data with top-tip lateral detection (standalone builder)."""
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

    # Map junctions + detect top-tip
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
        tip_threshold = int(primary_skel_len * TOP_TIP_PRIMARY_FRACTION)

        lateral_junctions = [lat["junction_end"] for lat in laterals]
        junction_indices = _find_junction_indices_on_primary(
            primary["skeleton"], lateral_junctions
        )
        primary["lateral_junction_indices"] = junction_indices

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
# CLEAN CUT (two half-planes)
# ===================================================================

def _get_local_direction(skeleton, idx, window=7):
    """Estimate local skeleton direction at *idx*."""
    start = max(0, idx - window)
    end = min(len(skeleton), idx + window + 1)
    if end - start < 2:
        return np.array([1, 0])
    pts = skeleton[start:end]
    d = pts[-1].astype(float) - pts[0].astype(float)
    norm = np.linalg.norm(d)
    return d / norm if norm > 1e-6 else np.array([1, 0])


def _compute_clean_cut_on_root(root_mask, skeleton, tid, center_idx, cut_length,
                               canvas_labels):
    """Low-level clean cut on a specific root mask and skeleton."""
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
    dot_s = (
        (points[:, 0] - pt_start[0]) * normal_start[0]
        + (points[:, 1] - pt_start[1]) * normal_start[1]
    )
    dot_e = (
        (points[:, 0] - pt_end[0]) * normal_end[0]
        + (points[:, 1] - pt_end[1]) * normal_end[1]
    )
    between = (dot_s >= 0) & (dot_e >= 0)
    erase = set(
        (int(ys_global[i]), int(xs_global[i])) for i in np.where(between)[0]
    )
    geo = {
        "pt_start": pt_start,
        "pt_end": pt_end,
        "dir_start": dir_start,
        "dir_end": dir_end,
        "start_idx": start_idx,
        "end_idx": end_idx,
    }
    return erase, geo


def _compute_clean_cut(root_data, center_idx, cut_length, canvas_labels):
    """Clean cut convenience wrapper using root_data dict."""
    return _compute_clean_cut_on_root(
        root_data["root_mask"],
        root_data["skeleton"],
        root_data["tid"],
        center_idx,
        cut_length,
        canvas_labels,
    )


# ===================================================================
# OBB
# ===================================================================

def _compute_obb_3cc(erase_pixels, cut_geometry, primary_mask, lateral_masks,
                     occluded_mask, stump_margin=5):
    """Compute surgically tight OBB for a 3cc cut using dilated stump detection."""
    if not erase_pixels or cut_geometry is None:
        return None, None

    combined = primary_mask.copy()
    for lm in lateral_masks:
        combined = np.maximum(combined, lm)

    h, w = combined.shape
    ec = np.array(list(erase_pixels))
    ey1, ex1 = ec.min(axis=0)
    ey2, ex2 = ec.max(axis=0)

    pad = stump_margin + 2
    y1, y2 = max(0, ey1 - pad), min(h, ey2 + pad + 1)
    x1, x2 = max(0, ex1 - pad), min(w, ex2 + pad + 1)

    local_erase = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    for py, px in erase_pixels:
        local_erase[py - y1, px - x1] = 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * stump_margin + 1, 2 * stump_margin + 1)
    )
    local_dilated = cv2.dilate(local_erase, kernel, iterations=1)

    local_occluded = occluded_mask[y1:y2, x1:x2]
    local_combined = combined[y1:y2, x1:x2]

    stump_roi = (local_occluded > 0) & (local_combined > 0) & (local_dilated > 0)
    sy_local, sx_local = np.where(stump_roi)

    if len(sy_local) == 0:
        return None, None

    syg = sy_local + y1
    sxg = sx_local + x1

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
    """Count connected components inside an OBB region."""
    if obb_box is None:
        return -1
    h, w = occluded_mask.shape
    bi = obb_box.astype(np.int32).reshape(-1, 1, 2)
    bm = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(bm, [bi], 255)
    ins = (bm > 0) & (occluded_mask > 0)
    nl, _ = cv2.connectedComponents(ins.astype(np.uint8), connectivity=8)
    return nl - 1


# ===================================================================
# VALIDATION & HELPERS
# ===================================================================

def _validate_ncc(erase_pixels, affected_pixels, expected_cc):
    """Validate that erasing pixels produces the expected CC count."""
    remaining = affected_pixels - erase_pixels
    if len(remaining) < 20:
        return False
    coords = np.array(list(remaining))
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)
    lm = np.zeros((max_y - min_y + 3, max_x - min_x + 3), dtype=np.uint8)
    for py, px in remaining:
        lm[py - min_y + 1, px - min_x + 1] = 255
    nl, _ = cv2.connectedComponents(lm, connectivity=8)
    return (nl - 1) == expected_cc


def _check_overlap_safety(erase_pixels, overlap_labels):
    """Return False if any erase pixel falls in the overlap region."""
    if overlap_labels is None:
        return True
    for py, px in erase_pixels:
        if overlap_labels[py, px] > 0:
            return False
    return True


def _find_isolated_junctions(root_data):
    """Find junctions that are isolated (far from tips and other junctions)."""
    jis = root_data.get("lateral_junction_indices", [])
    if not jis:
        return []
    skel = root_data["skeleton"]
    sl = len(skel)
    gt, gj = root_data["geo_tip"], root_data["geo_junction"]
    tp = max(MIN_SKELETON_DISTANCE_FROM_TIP, int(sl * 0.05))
    jp = max(MIN_SKELETON_DISTANCE_FROM_JUNCTION, int(sl * 0.03))
    iso = []
    for i, ji in enumerate(jis):
        pt = tuple(skel[ji])
        if gt.get(pt, 0) < tp or gj.get(pt, 0) < jp:
            continue
        if any(
            abs(ji - jis[j]) < MIN_JUNCTION_SEPARATION
            for j in range(len(jis))
            if j != i
        ):
            continue
        iso.append(ji)
    return iso


def _find_valid_positions_3cc(root_data, cut_length, occupied_ranges):
    """Find valid 3cc cut positions near isolated junctions."""
    skel = root_data["skeleton"]
    sl = len(skel)
    half = cut_length // 2
    iso = _find_isolated_junctions(root_data)
    if not iso:
        return []
    candidates = []
    for ji in iso:
        for off in range(-JUNCTION_TARGET_TOLERANCE, JUNCTION_TARGET_TOLERANCE + 1):
            idx = ji + off
            if idx < half or idx >= sl - half:
                continue
            cs, ce = idx - half, idx + half
            if any(
                not (
                    ce + MIN_GAP_BETWEEN_OCCLUSIONS < os_
                    or cs - MIN_GAP_BETWEEN_OCCLUSIONS > oe
                )
                for os_, oe in occupied_ranges
            ):
                continue
            candidates.append((idx, ji))
    return candidates


def _get_laterals_at_junction(root_data, roots, ji):
    """Return laterals whose junction is near skeleton index *ji*."""
    pid = root_data["plant_id"]
    jp = root_data["skeleton"][ji]
    lats = []
    for tid, rd in roots.items():
        if rd["plant_id"] != pid or rd["root_type"] != "lateral":
            continue
        d = np.sqrt(
            (rd["junction_end"][0] - jp[0]) ** 2
            + (rd["junction_end"][1] - jp[1]) ** 2
        )
        if d < 10:
            lats.append(rd)
    return lats


def _cleanup_lateral_fragments(erase_pixels, laterals, canvas_labels):
    """Remove small disconnected lateral fragments near the erase zone."""
    extra = set()
    extra_pairs = []
    ec = np.array(list(erase_pixels))
    if len(ec) == 0:
        return erase_pixels, extra_pairs

    h, w = canvas_labels.shape

    # Dynamic blast radius based on occlusion size
    ey_min, ex_min = ec.min(axis=0)
    ey_max, ex_max = ec.max(axis=0)
    cut_height = ey_max - ey_min
    cut_width = ex_max - ex_min

    cut_center_y = (ey_min + ey_max) / 2.0
    cut_center_x = (ex_min + ex_max) / 2.0

    blast_radius = (max(cut_height, cut_width) / 2.0) + 20.0

    # Immediate proximity cleanup
    pad = FRAGMENT_PROXIMITY_RADIUS + 2
    ly1, ly2 = max(0, ey_min - pad), min(h, ey_max + pad + 1)
    lx1, lx2 = max(0, ex_min - pad), min(w, ex_max + pad + 1)

    el = np.zeros((ly2 - ly1, lx2 - lx1), dtype=np.uint8)
    for py, px in erase_pixels:
        el[py - ly1, px - lx1] = 255

    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * FRAGMENT_PROXIMITY_RADIUS + 1,) * 2
    )
    pm = cv2.dilate(el, k, iterations=1)

    for lat in laterals:
        near = set()
        for py, px in lat["root_pixels"]:
            if ly1 <= py < ly2 and lx1 <= px < lx2:
                if pm[py - ly1, px - lx1] > 0 and (py, px) not in erase_pixels:
                    near.add((py, px))
        if near:
            extra |= near
            extra_pairs.append((near, lat["tid"]))

    ce = erase_pixels | extra

    # Disconnected small fragment cleanup (with blast radius constraint)
    for lat in laterals:
        rem = lat["root_pixels"] - ce
        if len(rem) < 5:
            lo = lat["root_pixels"] - ce
            if lo:
                extra |= lo
                extra_pairs.append((lo, lat["tid"]))
            continue

        co = np.array(list(rem))
        my, mx = co.min(axis=0)
        My, Mx = co.max(axis=0)
        lm = np.zeros((My - my + 3, Mx - mx + 3), dtype=np.uint8)
        cm = {}
        for py, px in rem:
            ly, lx = py - my + 1, px - mx + 1
            lm[ly, lx] = 255
            cm[(ly, lx)] = (py, px)

        nl, lab = cv2.connectedComponents(lm, connectivity=8)
        if nl - 1 <= 1:
            continue

        sz = np.bincount(lab.ravel())
        sz[0] = 0
        lg = np.argmax(sz)

        for lb in range(1, nl):
            if lb == lg or sz[lb] >= MIN_FRAGMENT_SIZE:
                continue

            fy, fx = np.where(lab == lb)
            frag_center_y = np.mean(fy) + my - 1
            frag_center_x = np.mean(fx) + mx - 1

            dist_to_cut = np.sqrt(
                (frag_center_y - cut_center_y) ** 2
                + (frag_center_x - cut_center_x) ** 2
            )

            if dist_to_cut <= blast_radius:
                fp = set()
                for y, x in zip(fy, fx):
                    if (y, x) in cm:
                        fp.add(cm[(y, x)])
                if fp:
                    extra |= fp
                    extra_pairs.append((fp, lat["tid"]))

    return erase_pixels | extra, extra_pairs


def _try_apply_multi_erase(canvas_mask, erase_list, canvas_labels):
    """Apply multiple erase operations (each with its own TID) to a mask."""
    result = canvas_mask.copy()
    h, w = result.shape
    for erase_pixels, tid in erase_list:
        for py, px in erase_pixels:
            if 0 <= py < h and 0 <= px < w:
                if canvas_labels[py, px] == tid:
                    result[py, px] = 0
    return result


# ===================================================================
# SAMPLE CUT SIZES
# ===================================================================

def _sample_3cc_cut_lengths(primary_rd, laterals, category, rng, force_big=False):
    """Sample primary and lateral cut lengths for a 3cc occlusion."""
    sc = SIZE_CONFIG_3CC.get(category, SIZE_CONFIG_3CC["medium"])
    pri_skel_len = len(primary_rd["skeleton"])
    pri_max = int(pri_skel_len * sc["max_primary_fraction"])

    if force_big:
        pri_lo, pri_hi = sc["big_primary_range"]
    else:
        pri_lo, pri_hi = sc["primary_cut_range"]

    pri_cut = int(rng.integers(max(15, pri_lo), min(pri_max, pri_hi) + 1))

    do_extension = rng.random() < PROB_LATERAL_EXTENSION
    lat_cut = 0
    if do_extension and laterals:
        lat_skel_len = min(len(l["skeleton"]) for l in laterals)
        lat_max = int(lat_skel_len * sc["max_lateral_fraction"])

        if force_big:
            lat_lo, lat_hi = sc["big_lateral_range"]
        else:
            lat_lo, lat_hi = sc["lateral_ext_range"]

        pri_ratio = pri_cut / max(1, pri_max)
        if pri_ratio > 0.6:
            lat_hi = int(lat_hi * 0.6)
        elif pri_ratio > 0.4:
            lat_hi = int(lat_hi * 0.8)

        lat_hi = min(lat_max, lat_hi)
        lat_lo = min(lat_lo, lat_hi)
        if lat_hi > lat_lo:
            lat_cut = int(rng.integers(lat_lo, lat_hi + 1))

    return pri_cut, lat_cut, do_extension


# ===================================================================
# TOP-TIP 3CC
# ===================================================================

def _find_top_tip_junctions(primary_rd, roots):
    """Find top-tip junctions with exactly one lateral (clean 3cc targets)."""
    pid = primary_rd["plant_id"]
    primary_skel = primary_rd["skeleton"]
    primary_skel_len = len(primary_skel)
    tip_threshold = int(primary_skel_len * TOP_TIP_PRIMARY_FRACTION)

    results = []
    for tid, rd in roots.items():
        if rd["plant_id"] != pid or rd["root_type"] != "lateral":
            continue
        if not rd.get("is_top_tip", False):
            continue
        jpt = rd["junction_end"]
        dists = np.sqrt(
            (primary_skel[:, 0] - jpt[0]) ** 2
            + (primary_skel[:, 1] - jpt[1]) ** 2
        )
        min_idx = np.argmin(dists)
        if dists[min_idx] < 10 and min_idx <= tip_threshold:
            other_lats = _get_laterals_at_junction(primary_rd, roots, min_idx)
            if len(other_lats) == 1:
                results.append((int(min_idx), rd))
    return results


def _try_place_top_tip_3cc(primary_rd, lateral_rd, junction_idx, roots,
                           canvas_labels, overlap_labels, current_mask,
                           tracker, category, rng):
    """Place a 3cc occlusion at a top-tip junction.

    Cuts the primary just below the top-tip junction to produce three
    connected components: primary tip stub, lateral body, and primary body.
    """
    pri_tid = primary_rd["tid"]
    lat_tid = lateral_rd["tid"]
    pri_skel = primary_rd["skeleton"]
    pri_skel_len = len(pri_skel)

    sc = SIZE_CONFIG_3CC.get(category, SIZE_CONFIG_3CC["medium"])
    pri_lo, pri_hi = sc["primary_cut_range"]
    pri_max = int(pri_skel_len * sc["max_primary_fraction"])
    pri_hi = min(pri_hi, pri_max)
    pri_lo = min(pri_lo, pri_hi)
    if pri_hi <= pri_lo:
        return None, current_mask
    pri_cut = int(rng.integers(pri_lo, pri_hi + 1))
    half = pri_cut // 2

    min_center = max(half, junction_idx + 2)
    max_center = min(
        pri_skel_len - half - 1,
        junction_idx + TOP_TIP_CUT_ZONE_BELOW_JUNCTION + half,
    )

    if min_center >= max_center:
        pri_cut = max(10, pri_cut // 2)
        half = pri_cut // 2
        min_center = max(half, junction_idx + 2)
        max_center = min(
            pri_skel_len - half - 1,
            junction_idx + TOP_TIP_CUT_ZONE_BELOW_JUNCTION + half,
        )
        if min_center >= max_center:
            return None, current_mask

    occupied = tracker.get_skel_occupied(pri_tid)
    candidates = []
    for idx in range(min_center, max_center + 1):
        cs, ce = idx - half, idx + half
        if any(
            not (
                ce + MIN_GAP_BETWEEN_OCCLUSIONS < os_
                or cs - MIN_GAP_BETWEEN_OCCLUSIONS > oe
            )
            for os_, oe in occupied
        ):
            continue
        candidates.append(idx)

    if not candidates:
        return None, current_mask

    rng.shuffle(candidates)

    # Lateral extension
    do_ext = rng.random() < PROB_LATERAL_EXTENSION
    lat_cut = 0
    if do_ext:
        lat_skel_len = len(lateral_rd["skeleton"])
        lat_max = int(lat_skel_len * sc["max_lateral_fraction"])
        lat_lo, lat_hi = sc["lateral_ext_range"]
        lat_hi = min(lat_max, lat_hi)
        lat_lo = min(lat_lo, lat_hi)
        if lat_hi > lat_lo:
            lat_cut = int(rng.integers(lat_lo, lat_hi + 1))

    for center_idx in candidates[:8]:
        erase_pri, cut_geo = _compute_clean_cut(
            primary_rd, center_idx, pri_cut, canvas_labels
        )
        if len(erase_pri) < 5:
            continue
        if not _check_overlap_safety(erase_pri, overlap_labels):
            continue

        laterals_here = [lateral_rd]
        erase_cleaned, frag_pairs = _cleanup_lateral_fragments(
            erase_pri, laterals_here, canvas_labels
        )
        all_erase = set(erase_cleaned)
        erase_pairs = [(erase_pri, pri_tid)]
        erase_pairs.extend(frag_pairs)

        if do_ext and lat_cut > 0:
            ls = lateral_rd["skeleton"]
            if len(ls) >= lat_cut + 5:
                le, _ = _compute_clean_cut_on_root(
                    lateral_rd["root_mask"], ls, lat_tid,
                    lat_cut // 2, lat_cut, canvas_labels,
                )
                if len(le) > 3:
                    all_erase |= le
                    erase_pairs.append((le, lat_tid))

        if not tracker.check(all_erase):
            continue

        affected = set(primary_rd["root_pixels"]) | set(lateral_rd["root_pixels"])
        if not _validate_ncc(all_erase, affected, 3):
            continue

        trial = _try_apply_multi_erase(current_mask, erase_pairs, canvas_labels)
        lat_masks = [lateral_rd["root_mask"]]
        obb, obb_n = _compute_obb_3cc(
            all_erase, cut_geo, primary_rd["root_mask"], lat_masks, trial
        )
        cc = _count_cc_in_obb(trial, obb)

        if cc != 3:
            continue

        tracker.commit(
            all_erase,
            tid=pri_tid,
            skel_range=(
                center_idx - half - MIN_GAP_BETWEEN_OCCLUSIONS,
                center_idx + half + MIN_GAP_BETWEEN_OCCLUSIONS,
            ),
        )

        occ_type = "3cc_top_tip_ext" if (do_ext and lat_cut > 0) else "3cc_top_tip"

        return {
            "tid": pri_tid,
            "plant_id": primary_rd["plant_id"],
            "root_type": "primary",
            "center_idx": int(center_idx),
            "primary_cut_length": int(pri_cut),
            "lateral_cut_length": int(lat_cut),
            "has_extension": do_ext and lat_cut > 0,
            "size_forced_big": False,
            "erase_pixels": all_erase,
            "cut_geometry": cut_geo,
            "obb_box": obb,
            "obb_norm": obb_n,
            "cc_in_obb": cc,
            "expected_cc": 3,
            "occlusion_type": occ_type,
            "is_top_tip": True,
            "top_tip_lateral_tid": int(lat_tid),
        }, trial

    return None, current_mask


# ===================================================================
# STANDARD 3CC PLACEMENT
# ===================================================================

def _try_place_single_3cc(primary_rd, roots, canvas_labels, overlap_labels,
                          current_mask, occupied, category, rng,
                          force_big=False, tracker=None):
    """Attempt one standard mid-body 3cc occlusion at an isolated junction."""
    tid = primary_rd["tid"]
    skel = primary_rd["skeleton"]

    laterals_by_junction = {}
    for ji in _find_isolated_junctions(primary_rd):
        lats = _get_laterals_at_junction(primary_rd, roots, ji)
        if lats:
            laterals_by_junction[ji] = lats

    if not laterals_by_junction:
        return None, current_mask

    junctions = list(laterals_by_junction.keys())
    rng.shuffle(junctions)

    for ji in junctions:
        laterals = laterals_by_junction[ji]
        pri_cut, lat_cut, do_ext = _sample_3cc_cut_lengths(
            primary_rd, laterals, category, rng, force_big=force_big
        )

        candidates = _find_valid_positions_3cc(primary_rd, pri_cut, occupied)
        candidates = [(idx, j) for idx, j in candidates if j == ji]
        if not candidates:
            if not force_big:
                for frac in [0.6, 0.4]:
                    smaller = max(15, int(pri_cut * frac))
                    candidates = _find_valid_positions_3cc(
                        primary_rd, smaller, occupied
                    )
                    candidates = [(idx, j) for idx, j in candidates if j == ji]
                    if candidates:
                        pri_cut = smaller
                        lat_cut = max(0, int(lat_cut * frac))
                        break
            if not candidates:
                continue

        rng.shuffle(candidates)

        for center_idx, target_ji in candidates[:8]:
            erase_pri, cut_geo = _compute_clean_cut(
                primary_rd, center_idx, pri_cut, canvas_labels
            )
            if len(erase_pri) < 5:
                continue
            if not _check_overlap_safety(erase_pri, overlap_labels):
                continue
            if tracker is not None and not tracker.check(erase_pri):
                continue

            erase_cleaned, frag_pairs = _cleanup_lateral_fragments(
                erase_pri, laterals, canvas_labels
            )
            all_erase = set(erase_cleaned)
            erase_pairs = [(erase_pri, tid)]
            erase_pairs.extend(frag_pairs)

            if do_ext and lat_cut > 0:
                pri_cut_pt = primary_rd["skeleton"][center_idx]

                for lat in laterals:
                    ls = lat["skeleton"]
                    if len(ls) < lat_cut + 5:
                        continue

                    dists = np.sqrt(
                        (ls[:, 0] - pri_cut_pt[0]) ** 2
                        + (ls[:, 1] - pri_cut_pt[1]) ** 2
                    )
                    lat_junc_idx = int(np.argmin(dists))

                    if dists[lat_junc_idx] > 20:
                        continue

                    if lat_junc_idx < len(ls) // 2:
                        lat_center_idx = min(len(ls) - 1, lat_junc_idx + lat_cut // 2)
                    else:
                        lat_center_idx = max(0, lat_junc_idx - lat_cut // 2)

                    le, _ = _compute_clean_cut_on_root(
                        lat["root_mask"], ls, lat["tid"],
                        lat_center_idx, lat_cut, canvas_labels,
                    )

                    if len(le) > 3:
                        all_erase |= le
                        erase_pairs.append((le, lat["tid"]))

            affected = set(primary_rd["root_pixels"])
            for lat in laterals:
                affected |= lat["root_pixels"]

            if not _validate_ncc(all_erase, affected, 3):
                continue

            trial = _try_apply_multi_erase(current_mask, erase_pairs, canvas_labels)
            lat_masks = [l["root_mask"] for l in laterals]
            obb, obb_n = _compute_obb_3cc(
                all_erase, cut_geo, primary_rd["root_mask"], lat_masks, trial
            )
            cc = _count_cc_in_obb(trial, obb)

            if cc != 3:
                continue

            half = pri_cut // 2
            occupied.append((
                center_idx - half - MIN_GAP_BETWEEN_OCCLUSIONS,
                center_idx + half + MIN_GAP_BETWEEN_OCCLUSIONS,
            ))

            if tracker is not None:
                tracker.commit(
                    all_erase,
                    tid=tid,
                    skel_range=(
                        center_idx - half - MIN_GAP_BETWEEN_OCCLUSIONS,
                        center_idx + half + MIN_GAP_BETWEEN_OCCLUSIONS,
                    ),
                )

            occ_type = "3cc_ext" if (do_ext and lat_cut > 0) else "3cc"
            if force_big:
                occ_type += "_big"

            return {
                "tid": tid,
                "plant_id": primary_rd["plant_id"],
                "root_type": "primary",
                "center_idx": int(center_idx),
                "primary_cut_length": int(pri_cut),
                "lateral_cut_length": int(lat_cut),
                "has_extension": do_ext and lat_cut > 0,
                "size_forced_big": force_big,
                "erase_pixels": all_erase,
                "cut_geometry": cut_geo,
                "obb_box": obb,
                "obb_norm": obb_n,
                "cc_in_obb": cc,
                "expected_cc": 3,
                "occlusion_type": occ_type,
                "is_top_tip": False,
                "top_tip_lateral_tid": None,
            }, trial

    return None, current_mask


# ===================================================================
# PLACE ALL 3CC ON A DISH
# ===================================================================

def place_3cc_occlusions(roots, canvas_labels, overlap_labels, category, rng,
                         target_count, deferred_big_queue=None,
                         tracker=None, current_mask=None):
    """Place 3cc occlusions on a dish.

    Parameters
    ----------
    roots : dict
        Root data structures.
    canvas_labels, overlap_labels : numpy.ndarray
        Label images.
    category : str
        Root category.
    rng : numpy.random.Generator
        Random number generator.
    target_count : int
        Target number of occlusions.
    deferred_big_queue : list or None
        Queue of deferred big-occlusion attempts.
    tracker : GlobalOcclusionTracker or None
        Optional pre-existing tracker.
    current_mask : numpy.ndarray or None
        Optional pre-occluded mask.

    Returns
    -------
    tuple of (list, numpy.ndarray, list)
        ``(all_occlusions, occluded_mask, deferred_big_queue)``
    """
    if current_mask is None:
        current_mask = (canvas_labels > 0).astype(np.uint8) * 255
    else:
        current_mask = current_mask.copy()

    all_occlusions = []

    if deferred_big_queue is None:
        deferred_big_queue = []

    h, w = canvas_labels.shape
    if tracker is None:
        tracker = GlobalOcclusionTracker((h, w))

    primaries = [
        (tid, rd)
        for tid, rd in roots.items()
        if rd["root_type"] == "primary" and rd.get("lateral_junction_indices")
    ]
    if not primaries:
        return all_occlusions, current_mask, deferred_big_queue

    # Collect top-tip junction targets
    all_top_tip_junctions = []
    for tid, rd in primaries:
        for ji, lat_rd in _find_top_tip_junctions(rd, roots):
            all_top_tip_junctions.append((rd, lat_rd, ji))

    rng.shuffle(primaries)
    occupied_per_root = defaultdict(list)

    # Phase 1: deferred big occlusions
    remaining_deferred = []
    for deferred in deferred_big_queue:
        placed = False
        for tid, rd in primaries:
            occ, current_mask = _try_place_single_3cc(
                rd, roots, canvas_labels, overlap_labels,
                current_mask, occupied_per_root[tid], category, rng,
                force_big=True, tracker=tracker,
            )
            if occ is not None:
                all_occlusions.append(occ)
                placed = True
                break
        if not placed:
            deferred["attempts_remaining"] -= 1
            if deferred["attempts_remaining"] > 0:
                remaining_deferred.append(deferred)
    deferred_big_queue.clear()
    deferred_big_queue.extend(remaining_deferred)

    # Phase 2: fill remaining target
    remaining_target = target_count - len(all_occlusions)

    for i in range(remaining_target):
        occ = None
        roll = rng.random()

        # Mode 1: top-tip 3cc
        if roll < TOP_TIP_3CC_PROBABILITY and all_top_tip_junctions:
            pri_rd, lat_rd, ji = all_top_tip_junctions[
                rng.integers(0, len(all_top_tip_junctions))
            ]
            occ, current_mask = _try_place_top_tip_3cc(
                pri_rd, lat_rd, ji, roots,
                canvas_labels, overlap_labels,
                current_mask, tracker, category, rng,
            )

        # Mode 2: standard junction 3cc
        if occ is None:
            attempt_big = rng.random() < 0.20
            for tid, rd in primaries:
                occ, current_mask = _try_place_single_3cc(
                    rd, roots, canvas_labels, overlap_labels,
                    current_mask, occupied_per_root[tid], category, rng,
                    force_big=attempt_big, tracker=tracker,
                )
                if occ is not None:
                    break

            if occ is None and attempt_big:
                deferred_big_queue.append(
                    {"attempts_remaining": BIG_OCCLUSION_MAX_DEFER}
                )
                for tid, rd in primaries:
                    occ, current_mask = _try_place_single_3cc(
                        rd, roots, canvas_labels, overlap_labels,
                        current_mask, occupied_per_root[tid], category, rng,
                        force_big=False, tracker=tracker,
                    )
                    if occ is not None:
                        break

        if occ is not None:
            all_occlusions.append(occ)

    return all_occlusions, current_mask, deferred_big_queue


# ===================================================================
# PROCESS SINGLE DISH (entry point)
# ===================================================================

def process_single_dish_3cc(mask_path, config_name, output_dir, target_count,
                            deferred_big_queue=None, seed=None):
    """Process one dish for 3cc occlusions.

    Standalone entry point that loads images, builds roots, places
    occlusions, and writes outputs to disk.

    Returns
    -------
    dict
        Statistics dictionary.
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

        if deferred_big_queue is None:
            deferred_big_queue = []

        occlusions, occluded_mask, deferred_big_queue = place_3cc_occlusions(
            roots, canvas_labels, overlap_labels, category, rng,
            target_count, deferred_big_queue,
        )

        os.makedirs(output_dir, exist_ok=True)

        Image.fromarray(occluded_mask).save(
            os.path.join(output_dir, f"{dish_id}_mask_occluded_3cc.png")
        )

        obb_lines = []
        for occ in occlusions:
            if occ["obb_norm"] is not None:
                bn = occ["obb_norm"]
                line = "1 " + " ".join(
                    f"{bn[i, 0]:.6f} {bn[i, 1]:.6f}" for i in range(4)
                )
                obb_lines.append(line)

        with open(
            os.path.join(output_dir, f"{dish_id}_obb_3cc.txt"), "w"
        ) as f:
            for line in obb_lines:
                f.write(line + "\n")

        occ_meta = {
            "dish_id": dish_id,
            "config": config_name,
            "category": category,
            "target_count": target_count,
            "placed_count": len(occlusions),
            "deferred_big_remaining": len(deferred_big_queue),
            "top_tip_count": sum(
                1 for o in occlusions if o.get("is_top_tip", False)
            ),
            "occlusions": [
                {
                    "tid": int(o["tid"]),
                    "plant_id": int(o["plant_id"]),
                    "primary_cut_length": o["primary_cut_length"],
                    "lateral_cut_length": o["lateral_cut_length"],
                    "has_extension": o["has_extension"],
                    "size_forced_big": o["size_forced_big"],
                    "cc_in_obb": o["cc_in_obb"],
                    "occlusion_type": o["occlusion_type"],
                    "is_top_tip": o.get("is_top_tip", False),
                    "top_tip_lateral_tid": o.get("top_tip_lateral_tid", None),
                    "obb_xy": (
                        [[float(p[0]), float(p[1])] for p in o["obb_box"]]
                        if o["obb_box"] is not None
                        else None
                    ),
                }
                for o in occlusions
            ],
            "size_config_used": SIZE_CONFIG_3CC[category],
            "generation_timestamp": datetime.now().isoformat(),
            "seed": int(seed),
        }

        with open(
            os.path.join(output_dir, f"{dish_id}_occlusion_3cc_meta.json"), "w"
        ) as f:
            json.dump(occ_meta, f, indent=2)

        return {
            "dish_id": dish_id,
            "success": True,
            "target": target_count,
            "placed": len(occlusions),
            "deferred_big": len(deferred_big_queue),
            "top_tip_placed": sum(
                1 for o in occlusions if o.get("is_top_tip", False)
            ),
            "types": (
                dict(
                    zip(
                        *np.unique(
                            [o["occlusion_type"] for o in occlusions],
                            return_counts=True,
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