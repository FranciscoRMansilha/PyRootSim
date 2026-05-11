"""
Four-connected-component (4cc) occlusion cuts.

Targets X-crossings (two roots overlapping) and bilateral junctions
(primary + two laterals at the same point). Strict cc=4 in oriented
bounding boxes (OBBs). Three placement modes:

1. **Top-tip bilateral** — dedicated placement at top-tip bilateral
   junctions where two top-tip laterals meet the primary tip.
2. **Crossing** — cuts at overlap clusters where two roots cross.
3. **Standard bilateral** — cuts at bilateral junctions on primaries
   with two laterals at the same point.

Uses physically anchored lateral cuts and a "Ring of Fire" stump-count
validator to guarantee exactly four entry/exit points.
"""

import numpy as np
import os
import json
import cv2
from PIL import Image
from collections import defaultdict
from datetime import datetime
from skimage.morphology import skeletonize

# ===================================================================
# SIZE CONFIG
# ===================================================================

SIZE_CONFIG_4CC = {
    "short": {"cross": (18, 35), "bilateral": (20, 40)},
    "medium": {"cross": (20, 50), "bilateral": (25, 55)},
    "long": {"cross": (25, 70), "bilateral": (30, 75)},
    "extra_long": {"cross": (30, 100), "bilateral": (35, 100)},
    "extra_long_12": {"cross": (30, 100), "bilateral": (35, 100)},
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

MIN_OVERLAP_CLUSTER_SIZE = 8
MIN_FRAGMENT_SIZE = 15
FRAGMENT_PROXIMITY_RADIUS = 5
MAX_SKEL_DIST = 30
TIP_MARGIN = 20
MIN_SKELETON_DISTANCE_FROM_TIP = 20
MIN_SKELETON_DISTANCE_FROM_JUNCTION = 15
MIN_JUNCTION_SEPARATION = 40
MIN_GLOBAL_DISTANCE = 12

# Top-tip 4cc
TOP_TIP_PRIMARY_FRACTION = 0.06
TOP_TIP_4CC_PROBABILITY = 0.35
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

    def commit(self, erase_pixels):
        """Register a placed occlusion."""
        erase_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        for py, px in erase_pixels:
            if 0 <= py < self.h and 0 <= px < self.w:
                erase_mask[py, px] = 255
        dilated = cv2.dilate(erase_mask, self.kernel, iterations=1)
        self.proximity_mask = np.maximum(self.proximity_mask, dilated)


# ===================================================================
# LABEL & SKELETON HELPERS
# ===================================================================

def _decode_label(val):
    """Decode a label pixel value into plant ID and root type."""
    if val == 0:
        return None
    pid = val // 100
    rem = val % 100
    if rem == 0:
        return {"plant_id": pid, "root_type": "primary", "lateral_id": None}
    return {"plant_id": pid, "root_type": "lateral", "lateral_id": rem}


def _get_skeleton_graph(skel_mask):
    """Build adjacency graph from a binary skeleton mask."""
    pts = np.argwhere(skel_mask)
    if len(pts) < 2:
        return None, None, None
    pts_set = set(map(tuple, pts))
    adj = defaultdict(list)
    for p in pts_set:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                n = (p[0] + dy, p[1] + dx)
                if n in pts_set:
                    adj[p].append(n)
    endpoints = [p for p in pts_set if len(adj[p]) == 1]
    return pts_set, adj, endpoints


def _order_skeleton(start, adj):
    """BFS traversal of skeleton from *start*."""
    ordered, visited = [], set()
    q = [start]
    while q:
        curr = q.pop(0)
        if curr in visited:
            continue
        visited.add(curr)
        ordered.append(curr)
        for n in adj[curr]:
            if n not in visited:
                q.append(n)
    return np.array(ordered)


def _validate_and_compute_strict_4cc_obb(erase_pixels, occluded_mask,
                                         center_y, center_x, max_cut_length):
    """Ring-of-Fire strict filter: require exactly 4 entry/exit stump groups.

    Returns
    -------
    tuple of (bool, numpy.ndarray or None, numpy.ndarray or None)
        ``(success, obb_box, obb_norm)``
    """
    if not erase_pixels:
        return False, None, None

    ec = np.array(list(erase_pixels))
    ey1, ex1 = ec.min(axis=0)
    ey2, ex2 = ec.max(axis=0)
    h, w = occluded_mask.shape

    # Sprawl / teleport protection
    max_radius_allowed = max_cut_length + 20
    dists = np.sqrt((ec[:, 0] - center_y) ** 2 + (ec[:, 1] - center_x) ** 2)
    if dists.max() > max_radius_allowed:
        return False, None, None

    pad = 5
    y1, y2 = max(0, ey1 - pad), min(h, ey2 + pad + 1)
    x1, x2 = max(0, ex1 - pad), min(w, ex2 + pad + 1)

    local_erase = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    for py, px in erase_pixels:
        local_erase[py - y1, px - x1] = 255

    # 2-pixel halo ("Ring of Fire")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    local_dilated = cv2.dilate(local_erase, kernel, iterations=1)
    halo = local_dilated - local_erase

    # Stumps: surviving pixels inside the halo
    local_occ = occluded_mask[y1:y2, x1:x2]
    stumps = (halo > 0) & (local_occ > 0)

    stumps_uint8 = stumps.astype(np.uint8) * 255
    nl, _ = cv2.connectedComponents(stumps_uint8, connectivity=8)

    if (nl - 1) != 4:
        return False, None, None

    # Surgically tight OBB
    sy_local, sx_local = np.where(stumps)
    syg = sy_local + y1
    sxg = sx_local + x1

    ay = np.concatenate([ec[:, 0], syg])
    ax = np.concatenate([ec[:, 1], sxg])

    cp = np.column_stack([ax, ay]).astype(np.float32)
    r = cv2.minAreaRect(cp)
    b = cv2.boxPoints(r)

    bn = b.copy()
    bn[:, 0] /= w
    bn[:, 1] /= h

    return True, b, bn


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
    s = sorted(endpoints, key=lambda p: p[0])
    return s[0], s[-1]


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

def _build_roots_4cc(canvas_labels, overlap_labels):
    """Build root data with top-tip detection (standalone builder for 4cc)."""
    unique_tids = np.unique(canvas_labels)
    unique_tids = unique_tids[unique_tids > 0]
    h, w = canvas_labels.shape
    roots = {}

    for tid in unique_tids:
        decoded = _decode_label(int(tid))
        if decoded is None:
            continue

        root_mask = (canvas_labels == tid).astype(np.uint8)
        if overlap_labels is not None:
            root_mask = np.maximum(root_mask, (overlap_labels == tid).astype(np.uint8))

        root_pixels = set(map(tuple, np.argwhere(root_mask > 0)))
        if len(root_pixels) < 50:
            continue

        skel = skeletonize(root_mask > 0)
        pts_set, adj, endpoints = _get_skeleton_graph(skel)
        if not pts_set or len(endpoints) < 2:
            continue

        if decoded["root_type"] == "primary":
            junction, tip = _identify_primary_endpoints(endpoints)
        else:
            junction, tip = _identify_lateral_endpoints(
                endpoints, canvas_labels, decoded["plant_id"]
            )

        if junction is None or tip is None:
            continue

        ordered = _order_skeleton(junction, adj)
        geo_junction = _compute_geodesic_distances(ordered)
        ordered_tip = _order_skeleton(tip, adj)
        geo_tip = _compute_geodesic_distances(ordered_tip)

        roots[int(tid)] = {
            "tid": int(tid),
            "plant_id": decoded["plant_id"],
            "root_type": decoded["root_type"],
            "lateral_id": decoded["lateral_id"],
            "root_pixels": root_pixels,
            "root_mask": root_mask,
            "skeleton": ordered,
            "tip_end": tip,
            "junction_end": junction,
            "geo_tip": geo_tip,
            "geo_junction": geo_junction,
            "lateral_junction_indices": [],
            "is_top_tip": False,
        }

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

        jis = []
        for lat in laterals:
            jpt = lat["junction_end"]
            dists = np.sqrt(
                (primary["skeleton"][:, 0] - jpt[0]) ** 2
                + (primary["skeleton"][:, 1] - jpt[1]) ** 2
            )
            mi = np.argmin(dists)
            if dists[mi] < 8:
                jis.append(int(mi))
            if dists[mi] < 10 and mi <= tip_threshold:
                lat["is_top_tip"] = True

        primary["lateral_junction_indices"] = jis

    return roots


# ===================================================================
# CROSSING CLUSTER DETECTION
# ===================================================================

def _find_crossing_clusters(canvas_labels, overlap_labels):
    """Find overlap clusters where two roots cross each other."""
    if overlap_labels is None:
        return []
    mask = (overlap_labels > 0).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clusters = []

    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_OVERLAP_CLUSTER_SIZE:
            continue
        pts = np.argwhere(labels == i)

        tids = set()
        for p in pts:
            py, px = p[0], p[1]
            val1 = int(canvas_labels[py, px])
            val2 = int(overlap_labels[py, px])
            if val1 > 0:
                tids.add(val1)
            if val2 > 0:
                tids.add(val2)

        if len(tids) >= 2:
            center_y = float(centroids[i][1])
            center_x = float(centroids[i][0])
            clusters.append({
                "center": np.array([center_y, center_x]),
                "tids": sorted(list(tids)),
                "pixels": set(map(tuple, pts)),
                "size": len(pts),
            })

    return sorted(clusters, key=lambda c: c["size"], reverse=True)


# ===================================================================
# CLEAN CUT
# ===================================================================

def _get_local_dir(skel, idx, window=7):
    """Estimate local skeleton direction at *idx*."""
    s = max(0, idx - window)
    e = min(len(skel) - 1, idx + window)
    d = skel[e].astype(float) - skel[s].astype(float)
    norm = np.linalg.norm(d)
    return d / norm if norm > 1e-6 else np.array([1, 0])


def _compute_clean_cut_4cc(rd, center_idx, length, canvas_labels):
    """Compute a clean two-half-plane cut for 4cc placement."""
    skel = rd["skeleton"]
    root_mask = rd["root_mask"]
    h, w = root_mask.shape
    half = length // 2
    idx1 = max(0, center_idx - half)
    idx2 = min(len(skel) - 1, center_idx + half)
    if idx2 <= idx1:
        return set()

    p1 = skel[idx1].astype(float)
    p2 = skel[idx2].astype(float)
    d1 = _get_local_dir(skel, idx1)
    d2 = _get_local_dir(skel, idx2)
    center_pt = skel[center_idx].astype(float)

    if np.dot(d1, center_pt - p1) < 0:
        d1 = -d1
    if np.dot(d2, center_pt - p2) < 0:
        d2 = -d2

    pts_roi = skel[idx1 : idx2 + 1]
    margin = 20
    y1 = max(0, int(pts_roi[:, 0].min()) - margin)
    y2 = min(h - 1, int(pts_roi[:, 0].max()) + margin)
    x1 = max(0, int(pts_roi[:, 1].min()) - margin)
    x2 = min(w - 1, int(pts_roi[:, 1].max()) + margin)

    roi_root = root_mask[y1 : y2 + 1, x1 : x2 + 1] > 0
    ys_l, xs_l = np.where(roi_root)
    if len(ys_l) == 0:
        return set()

    ys_g = ys_l + y1
    xs_g = xs_l + x1
    pts = np.column_stack([ys_g.astype(float), xs_g.astype(float)])

    dot1 = (pts[:, 0] - p1[0]) * d1[0] + (pts[:, 1] - p1[1]) * d1[1]
    dot2 = (pts[:, 0] - p2[0]) * d2[0] + (pts[:, 1] - p2[1]) * d2[1]
    inside = (dot1 >= 0) & (dot2 >= 0)

    return set((int(ys_g[i]), int(xs_g[i])) for i in np.where(inside)[0])


def _find_nearest_skeleton_idx(skeleton, center, max_dist=30):
    """Find the skeleton index nearest to *center*, within *max_dist*."""
    center = np.array(center, dtype=float)
    dists = np.sqrt(np.sum((skeleton.astype(float) - center) ** 2, axis=1))
    min_idx = np.argmin(dists)
    if dists[min_idx] <= max_dist:
        return int(min_idx)
    return None


# ===================================================================
# FRAGMENT CLEANUP
# ===================================================================

def _cleanup_fragments_general(erase_pixels, roots_involved, canvas_labels):
    """Remove small disconnected fragments near the erase zone."""
    extra = set()
    extra_pairs = []
    ec = np.array(list(erase_pixels))
    if len(ec) == 0:
        return erase_pixels, extra_pairs

    h, w = canvas_labels.shape
    ey1, ex1 = ec.min(axis=0)
    ey2, ex2 = ec.max(axis=0)
    pad = FRAGMENT_PROXIMITY_RADIUS + 2
    ly1, ly2 = max(0, ey1 - pad), min(h, ey2 + pad + 1)
    lx1, lx2 = max(0, ex1 - pad), min(w, ex2 + pad + 1)
    el = np.zeros((ly2 - ly1, lx2 - lx1), dtype=np.uint8)
    for py, px in erase_pixels:
        el[py - ly1, px - lx1] = 255
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * FRAGMENT_PROXIMITY_RADIUS + 1,) * 2
    )
    pm = cv2.dilate(el, k, iterations=1)

    for rd in roots_involved:
        near = set()
        for py, px in rd["root_pixels"]:
            if ly1 <= py < ly2 and lx1 <= px < lx2:
                if pm[py - ly1, px - lx1] > 0 and (py, px) not in erase_pixels:
                    near.add((py, px))
        if near:
            extra |= near
            extra_pairs.append((near, rd["tid"]))

    ce = erase_pixels | extra
    for rd in roots_involved:
        rem = rd["root_pixels"] - ce
        if len(rem) < 10:
            lo = rd["root_pixels"] - ce
            if lo:
                extra |= lo
                extra_pairs.append((lo, rd["tid"]))
            continue
        coords = np.array(list(rem))
        my, mx = coords.min(axis=0)
        My, Mx = coords.max(axis=0)
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
            fp = set()
            fy, fx = np.where(lab == lb)
            for y, x in zip(fy, fx):
                if (y, x) in cm:
                    fp.add(cm[(y, x)])
            if fp:
                extra |= fp
                extra_pairs.append((fp, rd["tid"]))

    return erase_pixels | extra, extra_pairs


# ===================================================================
# VALIDATION (Local Patch)
# ===================================================================

def _validate_ncc(erase_pixels, canvas_labels, padding=30, min_cc_size=10):
    """Validate topological change locally using a padded patch comparison."""
    if not erase_pixels:
        return False

    ec = np.array(list(erase_pixels))
    ey1, ex1 = ec.min(axis=0)
    ey2, ex2 = ec.max(axis=0)

    h, w = canvas_labels.shape
    y1, y2 = max(0, ey1 - padding), min(h, ey2 + padding + 1)
    x1, x2 = max(0, ex1 - padding), min(w, ex2 + padding + 1)

    local_mask = (canvas_labels[y1:y2, x1:x2] > 0).astype(np.uint8) * 255
    nl_before, _ = cv2.connectedComponents(local_mask, connectivity=8)

    for py, px in erase_pixels:
        if y1 <= py < y2 and x1 <= px < x2:
            local_mask[py - y1, px - x1] = 0

    nl_after, labels, stats, _ = cv2.connectedComponentsWithStats(
        local_mask, connectivity=8
    )

    delta = nl_after - nl_before
    if delta not in [2, 3]:
        return False

    for i in range(1, nl_after):
        if stats[i, cv2.CC_STAT_AREA] < min_cc_size:
            return False

    return True


# ===================================================================
# OBB (1D Skeleton generator)
# ===================================================================

def _compute_obb_4cc(erase_pixels, target_rds, canvas_shape):
    """Compute a surgically tight OBB using only 1D skeleton coordinates."""
    if not erase_pixels:
        return None, None

    h, w = canvas_shape
    skel_pts = []

    for rd in target_rds:
        skel = rd["skeleton"]
        in_erase = []

        for idx, pt in enumerate(skel):
            if (int(pt[0]), int(pt[1])) in erase_pixels:
                in_erase.append(idx)

        if not in_erase:
            continue

        for idx in in_erase:
            skel_pts.append(skel[idx])

        min_idx = in_erase[0]
        max_idx = in_erase[-1]

        start_cap = max(0, min_idx - 3)
        for i in range(start_cap, min_idx):
            skel_pts.append(skel[i])

        end_cap = min(len(skel), max_idx + 4)
        for i in range(max_idx + 1, end_cap):
            skel_pts.append(skel[i])

    if len(skel_pts) < 3:
        return None, None

    pts_array = np.array(skel_pts, dtype=np.float32)
    cp = np.column_stack([pts_array[:, 1], pts_array[:, 0]])
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
# BILATERAL JUNCTION DETECTION
# ===================================================================

def _find_bilateral_junctions(rd, roots):
    """Find pairs of nearby junction indices that form bilateral junctions."""
    jis = rd.get("lateral_junction_indices", [])
    if len(jis) < 2:
        return []
    skel = rd["skeleton"]
    sl = len(skel)
    gt, gj = rd["geo_tip"], rd["geo_junction"]
    tp = max(MIN_SKELETON_DISTANCE_FROM_TIP, int(sl * 0.05))
    jp = max(MIN_SKELETON_DISTANCE_FROM_JUNCTION, int(sl * 0.03))
    pid = rd["plant_id"]

    bilateral = []
    used = set()
    for i, ji_a in enumerate(jis):
        if i in used:
            continue
        for j, ji_b in enumerate(jis):
            if j <= i or j in used:
                continue
            if abs(ji_a - ji_b) < 15:
                mid = (ji_a + ji_b) // 2
                pt = tuple(skel[mid])
                if gt.get(pt, 0) < tp or gj.get(pt, 0) < jp:
                    continue
                jp_a = skel[ji_a]
                jp_b = skel[ji_b]
                lats = []
                for tid_l, r in roots.items():
                    if r["plant_id"] != pid or r["root_type"] != "lateral":
                        continue
                    jpt = r["junction_end"]
                    da = np.sqrt(
                        (jpt[0] - jp_a[0]) ** 2 + (jpt[1] - jp_a[1]) ** 2
                    )
                    db = np.sqrt(
                        (jpt[0] - jp_b[0]) ** 2 + (jpt[1] - jp_b[1]) ** 2
                    )
                    if da < 10 or db < 10:
                        lats.append(r)
                if len(lats) == 2:
                    bilateral.append((mid, lats))
                    used.add(i)
                    used.add(j)
                    break
    return bilateral


def _find_top_tip_bilateral(primary_rd, roots):
    """Find bilateral junctions among top-tip laterals."""
    pid = primary_rd["plant_id"]
    primary_skel = primary_rd["skeleton"]
    primary_skel_len = len(primary_skel)
    tip_threshold = int(primary_skel_len * TOP_TIP_PRIMARY_FRACTION)

    tt_laterals = []
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
        mi = np.argmin(dists)
        if dists[mi] < 10 and mi <= tip_threshold:
            tt_laterals.append((int(mi), rd))

    if len(tt_laterals) < 2:
        return []

    results = []
    used = set()
    for i in range(len(tt_laterals)):
        if i in used:
            continue
        for j in range(i + 1, len(tt_laterals)):
            if j in used:
                continue
            ji_a, lat_a = tt_laterals[i]
            ji_b, lat_b = tt_laterals[j]
            if abs(ji_a - ji_b) < 20:
                mid = (ji_a + ji_b) // 2
                results.append((mid, [lat_a, lat_b]))
                used.add(i)
                used.add(j)
                break

    return results


# ===================================================================
# PLACEMENT: TOP-TIP BILATERAL
# ===================================================================

def _try_place_top_tip_4cc_bilateral(primary_rd, mid_idx, laterals, roots,
                                     canvas_labels, current_mask, tracker,
                                     category, rng):
    """Attempt a 4cc cut at a top-tip bilateral junction."""
    pri_skel = primary_rd["skeleton"]
    pri_skel_len = len(pri_skel)

    len_range = SIZE_CONFIG_4CC.get(category, SIZE_CONFIG_4CC["medium"])["bilateral"]
    cut_length = int(rng.integers(len_range[0], len_range[1] + 1))
    half = cut_length // 2

    min_center = max(half, mid_idx + 2)
    max_center = min(
        pri_skel_len - half - 1,
        mid_idx + TOP_TIP_CUT_ZONE_BELOW_JUNCTION + half,
    )

    if min_center >= max_center:
        cut_length = max(10, cut_length // 2)
        half = cut_length // 2
        min_center = max(half, mid_idx + 2)
        max_center = min(
            pri_skel_len - half - 1,
            mid_idx + TOP_TIP_CUT_ZONE_BELOW_JUNCTION + half,
        )
        if min_center >= max_center:
            return None, current_mask

    candidates = list(range(min_center, max_center + 1))
    if not candidates:
        return None, current_mask
    rng.shuffle(candidates)

    for center_idx in candidates[:8]:
        erase_pri = _compute_clean_cut_4cc(
            primary_rd, center_idx, cut_length, canvas_labels
        )
        if len(erase_pri) < 5:
            continue

        all_erase_cleaned, frag_pairs = _cleanup_fragments_general(
            erase_pri, laterals, canvas_labels
        )
        all_erase = set(all_erase_cleaned)

        # Physically anchored lateral cuts
        pri_cut_pt = pri_skel[center_idx]
        for lat in laterals:
            ls = lat["skeleton"]
            if len(ls) > 15:
                dists = np.sqrt(
                    (ls[:, 0] - pri_cut_pt[0]) ** 2
                    + (ls[:, 1] - pri_cut_pt[1]) ** 2
                )
                lat_junc_idx = int(np.argmin(dists))

                if dists[lat_junc_idx] > 20:
                    continue

                ext = int(rng.integers(5, min(25, len(ls) // 3) + 1))

                if lat_junc_idx < len(ls) // 2:
                    lat_center_idx = min(len(ls) - 1, lat_junc_idx + ext // 2)
                else:
                    lat_center_idx = max(0, lat_junc_idx - ext // 2)

                le = _compute_clean_cut_4cc(lat, lat_center_idx, ext, canvas_labels)
                if len(le) > 3:
                    all_erase |= le

        if not tracker.check(all_erase):
            continue

        trial = current_mask.copy()
        h, w = trial.shape
        for py, px in all_erase:
            if 0 <= py < h and 0 <= px < w:
                trial[py, px] = 0

        max_cut = cut_length + 20
        cy, cx = pri_skel[center_idx]
        success, obb_box, obb_norm = _validate_and_compute_strict_4cc_obb(
            all_erase, trial, cy, cx, max_cut
        )

        if not success:
            continue

        tracker.commit(all_erase)
        return {
            "type": "4cc_top_tip_bilateral",
            "erase": all_erase,
            "tids": [int(primary_rd["tid"])] + [int(l["tid"]) for l in laterals],
            "plant_id_a": int(primary_rd["plant_id"]),
            "plant_id_b": int(primary_rd["plant_id"]),
            "center": [int(cy), int(cx)],
            "cut_length_a": cut_length,
            "cut_length_b": 0,
            "overlap_size": 0,
            "cc_in_obb": 4,
            "expected_cc": 4,
            "obb_box": obb_box,
            "obb_norm": obb_norm,
            "is_top_tip": True,
        }, trial

    return None, current_mask


# ===================================================================
# PLACEMENT: CROSSING 4CC
# ===================================================================

def _try_place_4cc_crossing(clusters, roots, canvas_labels, current_mask,
                            tracker, rng, category):
    """Attempt a 4cc cut at an overlap crossing cluster."""
    placed = []
    len_range = SIZE_CONFIG_4CC.get(category, SIZE_CONFIG_4CC["medium"])["cross"]

    for cluster in clusters:
        valid_rds = [roots[t] for t in cluster["tids"] if t in roots]
        if len(valid_rds) < 2:
            continue
        rd_a, rd_b = valid_rds[0], valid_rds[1]

        idx_a = _find_nearest_skeleton_idx(rd_a["skeleton"], cluster["center"])
        idx_b = _find_nearest_skeleton_idx(rd_b["skeleton"], cluster["center"])
        if idx_a is None or idx_b is None:
            continue

        sa, sb = len(rd_a["skeleton"]), len(rd_b["skeleton"])
        if idx_a < TIP_MARGIN or idx_a > sa - TIP_MARGIN:
            continue
        if idx_b < TIP_MARGIN or idx_b > sb - TIP_MARGIN:
            continue

        for _ in range(5):
            cut_a = int(rng.integers(len_range[0], len_range[1] + 1))
            cut_b = int(rng.integers(len_range[0], len_range[1] + 1))

            erase_a = _compute_clean_cut_4cc(rd_a, idx_a, cut_a, canvas_labels)
            erase_b = _compute_clean_cut_4cc(rd_b, idx_b, cut_b, canvas_labels)
            if len(erase_a) < 5 or len(erase_b) < 5:
                continue

            all_erase = erase_a | erase_b
            all_erase_cleaned, _ = _cleanup_fragments_general(
                all_erase, [rd_a, rd_b], canvas_labels
            )

            if not tracker.check(all_erase_cleaned):
                continue

            trial = current_mask.copy()
            h, w = trial.shape
            for py, px in all_erase_cleaned:
                if 0 <= py < h and 0 <= px < w:
                    trial[py, px] = 0

            max_cut = max(cut_a, cut_b)
            cy, cx = cluster["center"]
            success, obb_box, obb_norm = _validate_and_compute_strict_4cc_obb(
                all_erase_cleaned, trial, cy, cx, max_cut
            )

            if not success:
                continue

            tracker.commit(all_erase_cleaned)
            placed.append({
                "type": "4cc_cross",
                "erase": all_erase_cleaned,
                "tids": [int(rd_a["tid"]), int(rd_b["tid"])],
                "plant_id_a": int(rd_a["plant_id"]),
                "plant_id_b": int(rd_b["plant_id"]),
                "center": [float(cy), float(cx)],
                "cut_length_a": cut_a,
                "cut_length_b": cut_b,
                "overlap_size": int(cluster["size"]),
                "cc_in_obb": 4,
                "expected_cc": 4,
                "obb_box": obb_box,
                "obb_norm": obb_norm,
                "is_top_tip": False,
            })
            current_mask = trial
            break

        if placed:
            break

    return placed, current_mask


# ===================================================================
# PLACEMENT: STANDARD BILATERAL 4CC
# ===================================================================

def _try_place_4cc_bilateral(roots, canvas_labels, current_mask, tracker,
                             rng, category):
    """Attempt a 4cc cut at a standard bilateral junction."""
    placed = []
    len_range = SIZE_CONFIG_4CC.get(category, SIZE_CONFIG_4CC["medium"])["bilateral"]

    primaries = [
        (tid, rd)
        for tid, rd in roots.items()
        if rd["root_type"] == "primary"
        and len(rd.get("lateral_junction_indices", [])) >= 2
    ]

    for tid, rd in primaries:
        bilaterals = _find_bilateral_junctions(rd, roots)
        if not bilaterals:
            continue

        skel = rd["skeleton"]
        sl = len(skel)

        for mid_idx, lats in bilaterals:
            cut_length = int(rng.integers(len_range[0], len_range[1] + 1))
            half = cut_length // 2
            if mid_idx - half < 0 or mid_idx + half >= sl:
                continue

            erase_pri = _compute_clean_cut_4cc(rd, mid_idx, cut_length, canvas_labels)
            if len(erase_pri) < 5:
                continue

            all_erase_cleaned, frag_pairs = _cleanup_fragments_general(
                erase_pri, lats, canvas_labels
            )
            all_erase = set(all_erase_cleaned)

            # Physically anchored lateral cuts
            pri_cut_pt = skel[mid_idx]
            for lat in lats:
                ls = lat["skeleton"]
                if len(ls) > 15:
                    dists = np.sqrt(
                        (ls[:, 0] - pri_cut_pt[0]) ** 2
                        + (ls[:, 1] - pri_cut_pt[1]) ** 2
                    )
                    lat_junc_idx = int(np.argmin(dists))

                    if dists[lat_junc_idx] > 20:
                        continue

                    ext = int(rng.integers(5, min(25, len(ls) // 3) + 1))

                    if lat_junc_idx < len(ls) // 2:
                        lat_center_idx = min(len(ls) - 1, lat_junc_idx + ext // 2)
                    else:
                        lat_center_idx = max(0, lat_junc_idx - ext // 2)

                    le = _compute_clean_cut_4cc(
                        lat, lat_center_idx, ext, canvas_labels
                    )
                    if len(le) > 3:
                        all_erase |= le

            if not tracker.check(all_erase):
                continue

            trial = current_mask.copy()
            h, w = trial.shape
            for py, px in all_erase:
                if 0 <= py < h and 0 <= px < w:
                    trial[py, px] = 0

            max_cut = cut_length + 20
            cy, cx = skel[mid_idx]
            success, obb_box, obb_norm = _validate_and_compute_strict_4cc_obb(
                all_erase, trial, cy, cx, max_cut
            )

            if not success:
                continue

            tracker.commit(all_erase)
            placed.append({
                "type": "4cc_bilateral",
                "erase": all_erase,
                "tids": [int(rd["tid"])] + [int(l["tid"]) for l in lats],
                "plant_id_a": int(rd["plant_id"]),
                "plant_id_b": int(rd["plant_id"]),
                "center": [int(cy), int(cx)],
                "cut_length_a": cut_length,
                "cut_length_b": 0,
                "overlap_size": 0,
                "cc_in_obb": 4,
                "expected_cc": 4,
                "obb_box": obb_box,
                "obb_norm": obb_norm,
                "is_top_tip": False,
            })
            current_mask = trial
            break

        if placed:
            break

    return placed, current_mask


# ===================================================================
# MAIN: PLACE ALL 4CC ON A DISH
# ===================================================================

def place_4cc_occlusions(roots, canvas_labels, overlap_labels, category, rng,
                         target_count=1, tracker=None, current_mask=None):
    """Place 4cc occlusions on a dish.

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
        Target number of occlusions (default 1).
    tracker : GlobalOcclusionTracker or None
        Optional pre-existing tracker.
    current_mask : numpy.ndarray or None
        Optional pre-occluded mask.

    Returns
    -------
    tuple of (list, numpy.ndarray)
        ``(all_placed, occluded_mask)``
    """
    if current_mask is None:
        current_mask = (canvas_labels > 0).astype(np.uint8) * 255
    else:
        current_mask = current_mask.copy()

    all_placed = []
    h, w = canvas_labels.shape
    if tracker is None:
        tracker = GlobalOcclusionTracker((h, w))

    primaries = [
        (tid, rd) for tid, rd in roots.items() if rd["root_type"] == "primary"
    ]

    all_tt_bilateral = []
    for tid, rd in primaries:
        for mid_idx, lats in _find_top_tip_bilateral(rd, roots):
            all_tt_bilateral.append((rd, mid_idx, lats))

    clusters = _find_crossing_clusters(canvas_labels, overlap_labels)

    for attempt in range(target_count * 3):
        if len(all_placed) >= target_count:
            break

        roll = rng.random()

        if roll < TOP_TIP_4CC_PROBABILITY and all_tt_bilateral:
            pri_rd, mid_idx, lats = all_tt_bilateral[
                rng.integers(0, len(all_tt_bilateral))
            ]
            occ, current_mask = _try_place_top_tip_4cc_bilateral(
                pri_rd, mid_idx, lats, roots, canvas_labels,
                current_mask, tracker, category, rng,
            )
            if occ is not None:
                all_placed.append(occ)
                continue

        if clusters:
            placed_list, current_mask = _try_place_4cc_crossing(
                clusters, roots, canvas_labels, current_mask, tracker, rng, category
            )
            if placed_list:
                all_placed.extend(placed_list)
                continue

        placed_list, current_mask = _try_place_4cc_bilateral(
            roots, canvas_labels, current_mask, tracker, rng, category
        )
        if placed_list:
            all_placed.extend(placed_list)

    return all_placed, current_mask


# ===================================================================
# PROCESS SINGLE DISH (entry point)
# ===================================================================

def process_single_dish_4cc(mask_path, config_name, output_dir,
                            target_count=1, seed=None):
    """Process one dish for 4cc occlusions.

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

        if not os.path.exists(labels_path):
            return {"dish_id": dish_id, "success": False, "error": "missing_files"}

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

        roots = _build_roots_4cc(canvas_labels, overlap_labels)

        occlusions, occluded_mask = place_4cc_occlusions(
            roots, canvas_labels, overlap_labels, category, rng, target_count
        )

        os.makedirs(output_dir, exist_ok=True)

        Image.fromarray(occluded_mask).save(
            os.path.join(output_dir, f"{dish_id}_mask_occluded_4cc.png")
        )

        obb_lines = []
        for occ in occlusions:
            if occ["obb_norm"] is not None:
                bn = occ["obb_norm"]
                line = "2 " + " ".join(
                    f"{bn[i, 0]:.6f} {bn[i, 1]:.6f}" for i in range(4)
                )
                obb_lines.append(line)

        with open(
            os.path.join(output_dir, f"{dish_id}_obb_4cc.txt"), "w"
        ) as f:
            for line in obb_lines:
                f.write(line + "\n")

        occ_meta = {
            "dish_id": dish_id,
            "config": config_name,
            "category": category,
            "target_count": target_count,
            "placed_count": len(occlusions),
            "top_tip_count": sum(
                1 for o in occlusions if o.get("is_top_tip", False)
            ),
            "occlusions": [
                {
                    "type": o["type"],
                    "tids": o["tids"],
                    "plant_id_a": o["plant_id_a"],
                    "plant_id_b": o["plant_id_b"],
                    "cut_length_a": o["cut_length_a"],
                    "cut_length_b": o["cut_length_b"],
                    "overlap_size": o["overlap_size"],
                    "cc_in_obb": o["cc_in_obb"],
                    "is_top_tip": o.get("is_top_tip", False),
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
            os.path.join(output_dir, f"{dish_id}_occlusion_4cc_meta.json"), "w"
        ) as f:
            json.dump(occ_meta, f, indent=2)

        return {
            "dish_id": dish_id,
            "success": True,
            "target": target_count,
            "placed": len(occlusions),
            "top_tip_placed": sum(
                1 for o in occlusions if o.get("is_top_tip", False)
            ),
            "types": (
                dict(
                    zip(
                        *np.unique(
                            [o["type"] for o in occlusions], return_counts=True
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