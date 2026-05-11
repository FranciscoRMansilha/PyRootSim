"""
Vectorised root data builder for discontinuity injection.

Called once per dish to construct root data structures (skeleton graphs,
geodesic distance maps, junction metadata) used by all cut modules.
Pixel operations are vectorised — no Python-level pixel loops for
mask construction.
"""

import numpy as np
import cv2
from collections import defaultdict
from skimage.morphology import skeletonize

# Fraction of the primary skeleton length used to detect top-tip laterals.
TOP_TIP_PRIMARY_FRACTION = 0.06


def decode_label(val):
    """Decode a label-image integer into plant ID and root type.

    Parameters
    ----------
    val : int
        Pixel value from the label image. ``0`` means background.

    Returns
    -------
    dict or None
        ``None`` for background. Otherwise a dict with keys
        ``plant_id``, ``root_type`` (``"primary"`` or ``"lateral"``),
        and ``lateral_id``.
    """
    if val == 0:
        return None
    plant_id = val // 100
    remainder = val % 100
    if remainder == 0:
        return {"plant_id": plant_id, "root_type": "primary", "lateral_id": None}
    return {"plant_id": plant_id, "root_type": "lateral", "lateral_id": remainder}


def build_roots_fast(canvas_labels, overlap_labels):
    """Build root data structures once per dish (vectorised).

    Constructs skeleton graphs, geodesic distance maps, and junction
    metadata for every root in the dish. The result is passed to all
    cut modules.

    Parameters
    ----------
    canvas_labels : numpy.ndarray
        2-D integer label image from the petri dish composer.
    overlap_labels : numpy.ndarray or None
        Optional overlap label image for roots that cross each other.

    Returns
    -------
    dict
        Keyed by integer TID. Each value is a dict containing masks,
        pixel sets, skeleton arrays, geodesic distances, junction
        indices, and a ``is_top_tip`` flag.
    """
    unique_tids = np.unique(canvas_labels)
    unique_tids = unique_tids[unique_tids > 0]
    h, w = canvas_labels.shape
    roots = {}

    for tid in unique_tids:
        decoded = decode_label(int(tid))
        if decoded is None:
            continue

        # Vectorized mask construction
        root_mask = (canvas_labels == tid).astype(np.uint8)
        if overlap_labels is not None:
            root_mask = np.maximum(root_mask, (overlap_labels == tid).astype(np.uint8))

        pixel_count = np.count_nonzero(root_mask)
        if pixel_count < 50:
            continue

        # Vectorized pixel coordinate extraction
        coords = np.argwhere(root_mask > 0)
        root_pixels_array = coords  # Ny2 array, kept as numpy
        root_pixels_set = set(map(tuple, coords))  # needed for set ops in validation

        # Single skeletonize call
        skel_mask = skeletonize(root_mask > 0)
        skel_pts = np.argwhere(skel_mask)
        if len(skel_pts) < 2:
            continue

        # Build adjacency with numpy acceleration
        pts_set, adj, endpoints = _fast_skeleton_graph(skel_pts)
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

        ordered = _order_skeleton(junction, adj, pts_set)
        geo_junction = _geodesic_distances_fast(ordered)
        ordered_tip = _order_skeleton(tip, adj, pts_set)
        geo_tip = _geodesic_distances_fast(ordered_tip)

        roots[int(tid)] = {
            "tid": int(tid),
            "plant_id": decoded["plant_id"],
            "root_type": decoded["root_type"],
            "lateral_id": decoded["lateral_id"],
            "root_pixels": root_pixels_set,
            "root_pixels_array": root_pixels_array,
            "root_mask": root_mask,
            "pixel_count": pixel_count,
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

    # Map junctions + top-tip detection
    _map_junctions_and_top_tip(roots)

    return roots


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fast_skeleton_graph(skel_pts):
    """Build skeleton adjacency using set lookups."""
    pts_set = set(map(tuple, skel_pts))
    adj = defaultdict(list)
    for p in pts_set:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                n = (p[0] + dy, p[1] + dx)
                if n in pts_set:
                    adj[p].append(n)
    endpoints = [p for p in pts_set if len(adj[p]) == 1]
    return pts_set, adj, endpoints


def _order_skeleton(start, adj, pts_set):
    """BFS traversal of skeleton from *start*, returning ordered point array."""
    ordered = []
    visited = set()
    queue = [start]
    while queue:
        curr = queue.pop(0)
        if curr in visited:
            continue
        visited.add(curr)
        ordered.append(curr)
        for n in adj[curr]:
            if n not in visited:
                queue.append(n)
    return np.array(ordered)


def _geodesic_distances_fast(ordered):
    """Vectorised geodesic distance computation along an ordered skeleton."""
    if len(ordered) < 2:
        return {tuple(ordered[0]): 0.0} if len(ordered) == 1 else {}
    diffs = np.diff(ordered.astype(np.float64), axis=0)
    steps = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    return {tuple(ordered[i]): float(cumulative[i]) for i in range(len(ordered))}


def _identify_primary_endpoints(endpoints):
    """Return (junction, tip) for a primary root — topmost is junction."""
    if len(endpoints) < 2:
        return None, None
    s = sorted(endpoints, key=lambda p: p[0])
    return s[0], s[-1]


def _identify_lateral_endpoints(endpoints, canvas_labels, plant_id):
    """Return (junction, tip) for a lateral root based on proximity to primary."""
    if len(endpoints) < 2:
        return None, None
    primary_tid = plant_id * 100
    h, w = canvas_labels.shape

    # Vectorized neighborhood check
    junction, tip = None, None
    for ep in endpoints:
        py, px = ep
        y1 = max(0, py - 3)
        y2 = min(h, py + 4)
        x1 = max(0, px - 3)
        x2 = min(w, px + 4)
        if np.any(canvas_labels[y1:y2, x1:x2] == primary_tid):
            junction = ep
        else:
            tip = ep

    if junction is None or tip is None:
        s = sorted(endpoints, key=lambda p: p[0])
        junction, tip = s[0], s[-1]
    return junction, tip


def _map_junctions_and_top_tip(roots):
    """Map lateral junctions onto primaries and detect top-tip laterals."""
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

        pri_skel = primary["skeleton"]
        pri_skel_len = len(pri_skel)
        tip_threshold = int(pri_skel_len * TOP_TIP_PRIMARY_FRACTION)

        # Vectorized junction mapping
        jis = []
        for lat in laterals:
            jpt = np.array(lat["junction_end"], dtype=np.float64)
            dists = np.sqrt(
                (pri_skel[:, 0].astype(np.float64) - jpt[0]) ** 2
                + (pri_skel[:, 1].astype(np.float64) - jpt[1]) ** 2
            )
            mi = np.argmin(dists)
            if dists[mi] < 8:
                jis.append(int(mi))
            if dists[mi] < 10 and mi <= tip_threshold:
                lat["is_top_tip"] = True

        primary["lateral_junction_indices"] = jis