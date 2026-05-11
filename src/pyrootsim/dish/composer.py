"""
Petri dish composer for PyRootSim.

Composes synthetic petri-dish images by placing multiple root systems
(primary + laterals) from RSA orchestrator output onto a shared canvas.
The default layout — a 2738 × 2535 canvas with 5 seedling placement
boxes — was designed to simulate the petri-dish format used by NPEC's
HADES high-throughput root phenotyping system, which images up to 5
*Arabidopsis thaliana* seedlings per dish. Supports sequential or
multiprocessing execution.

The canvas dimensions, box geometry, placement distributions, and
mixing probabilities are currently defined as module-level constants.
In a future version these will be wrapped in a configuration class
(similar to :mod:`pyrootsim.roots.configs`) so that users can easily
adapt the composer to different experimental setups without modifying
the source code.

Input structure (from the orchestrator)::

    rsa_dir/{primary_config}/{lateral_mode}/{root_id}_full_mask.png
    rsa_dir/{primary_config}/{lateral_mode}/{root_id}_full_labels.png
    rsa_dir/{primary_config}/{lateral_mode}/{root_id}_full.json

Output structure::

    output_dir/{dish_id}_mask.png
    output_dir/{dish_id}_labels.png
    output_dir/{dish_id}_overlap.png
    output_dir/{dish_id}_metadata.json
    output_dir/generation_summary.json
    output_dir/dish_metadata.csv

Usage::

    from pyrootsim.dish.composer import generate_all_dishes

    summary = generate_all_dishes(
        rsa_input_dir="RSA_dataset",
        output_dir="petri_dishes",
        num_dishes=1000,
        num_workers=4,
    )
"""

import os
import glob
import json
import shutil
import csv
import multiprocessing as mp
from datetime import datetime
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm


# =========================================================================
# Canvas & Box Configuration
# =========================================================================

CANVAS_CONFIG = {"width": 2738, "height": 2535}

BOX_CONFIG = {
    "num_boxes": 5,
    "box_width": 390,
    "box_height": 290,
    "box_y_start": 290,
}

DISTRIBUTION_CONFIG = {
    "type": "mixture",
    "gaussian_sigma": 0.25,
    "mixture_gaussian_weight": 0.80,
}

BOUNDARY_MARGIN = 45
EDGE_BOXES = [0, 4]

SKIP_CONFIG = {
    "skip_probability": 0.08,
    "min_plants_per_dish": 1,
}

MAX_PLACEMENT_ATTEMPTS = 20


# =========================================================================
# Size / Style / Lateral Mixing Configuration
# =========================================================================

CONFIGS_BY_CATEGORY = {
    "short": [
        "01_Short_Kinky_Noisy", "02_Short_Smooth_Clean",
        "03_Short_Kinky_Smooth",
    ],
    "medium": [
        "04_Medium_Kinky_Noisy", "05_Medium_Smooth_Snake",
        "06_Medium_Clean_GroundTruth",
    ],
    "long": [
        "07_Long_Kinky_Noisy", "08_Long_Sweeping_Curves",
        "09_Long_Smooth_Static",
    ],
    "extra_long": ["10_ExtraLong_Hybrid", "11_ExtraLong_Curvy_Clean"],
    "extra_long_12": ["12_ExtraLong_Mixed_Sweeping"],
}

CONFIG_TO_CATEGORY = {}
for _cat, _cfgs in CONFIGS_BY_CATEGORY.items():
    for _cfg in _cfgs:
        CONFIG_TO_CATEGORY[_cfg] = _cat

SIZE_MIX_CONFIG = {
    "homogeneous_prob": 0.70,
    "mix_1_prob": 0.20,
    "mix_2_prob": 0.08,
    "mix_3_prob": 0.02,
    "adjacency": {
        "short": ["medium"],
        "medium": ["short", "long"],
        "long": ["medium", "extra_long"],
        "extra_long": ["long", "extra_long_12"],
        "extra_long_12": ["extra_long"],
    },
}

STYLE_MIX_CONFIG = {
    "homogeneous_prob": 0.60,
    "styles_by_size": {
        "short": [
            "01_Short_Kinky_Noisy", "02_Short_Smooth_Clean",
            "03_Short_Kinky_Smooth",
        ],
        "medium": [
            "04_Medium_Kinky_Noisy", "05_Medium_Smooth_Snake",
            "06_Medium_Clean_GroundTruth",
        ],
        "long": [
            "07_Long_Kinky_Noisy", "08_Long_Sweeping_Curves",
            "09_Long_Smooth_Static",
        ],
        "extra_long": ["10_ExtraLong_Hybrid", "11_ExtraLong_Curvy_Clean"],
        "extra_long_12": ["12_ExtraLong_Mixed_Sweeping"],
    },
}

LATERAL_MIX_CONFIG = {
    "homogeneous_prob": 0.50,
    "compatible_groups": {
        "few": [
            "A_few_small", "B_few_horizontal", "C_few_arched", "D_few_mixed",
        ],
        "medium_count": [
            "E_medium_small", "F_medium_horizontal",
            "G_medium_arched", "H_medium_mixed",
        ],
        "many": [
            "I_many_small", "J_many_horizontal",
            "K_many_arched", "L_many_mixed",
        ],
    },
}


# =========================================================================
# Label Encoding
# =========================================================================

def encode_label(plant_id, lateral_id=None):
    """Encode plant + root identity into a single uint16 label.

    ``plant_id * 100 + 0`` → primary root.
    ``plant_id * 100 + lateral_id`` → lateral (lateral_id ≥ 1).
    """
    if lateral_id is None or lateral_id == 0:
        return plant_id * 100
    return plant_id * 100 + lateral_id


def decode_label(label_value):
    """Decode a uint16 label into plant ID and root type."""
    if label_value == 0:
        return None
    plant_id = label_value // 100
    remainder = label_value % 100
    if remainder == 0:
        return {
            "plant_id": plant_id, "root_type": "primary", "lateral_id": None,
        }
    return {
        "plant_id": plant_id, "root_type": "lateral",
        "lateral_id": remainder,
    }


# =========================================================================
# Geometry Helpers
# =========================================================================

def _calculate_boxes():
    """Compute bounding boxes for the plant placement grid."""
    cw = CANVAS_CONFIG["width"]
    nb = BOX_CONFIG["num_boxes"]
    bw = BOX_CONFIG["box_width"]
    bh = BOX_CONFIG["box_height"]
    by = BOX_CONFIG["box_y_start"]
    strip = cw / nb
    return [
        {
            "box_id": i,
            "x1": i * strip + strip / 2 - bw / 2,
            "y1": by,
            "x2": i * strip + strip / 2 + bw / 2,
            "y2": by + bh,
            "width": bw,
            "height": bh,
        }
        for i in range(nb)
    ]


def _sample_placement(box, rng):
    """Sample a (x, y) position within a placement box."""
    sigma = DISTRIBUTION_CONFIG["gaussian_sigma"]
    gw = DISTRIBUTION_CONFIG["mixture_gaussian_weight"]
    if rng.random() < gw:
        xn = np.clip(rng.normal(0.5, sigma), 0, 1)
        yn = np.clip(rng.normal(0.5, sigma), 0, 1)
    else:
        xn, yn = rng.random(), rng.random()
    return box["x1"] + xn * box["width"], box["y1"] + yn * box["height"]


def _compute_boundary_correction(primary_xs, offset_x, canvas_w, margin):
    """Compute a horizontal shift to keep the primary root within bounds."""
    cx = primary_xs + offset_x
    if cx.min() < 0:
        return int(abs(cx.min()) + margin)
    if cx.max() >= canvas_w:
        return int(-(cx.max() - canvas_w + 1 + margin))
    return 0


def _check_primary_collision(primary_canvas, prim_ys, prim_xs,
                             off_y, off_x, ch, cw):
    """Check whether a primary root would overlap existing primaries."""
    cy = prim_ys + off_y
    cx = prim_xs + off_x
    valid = (cy >= 0) & (cy < ch) & (cx >= 0) & (cx < cw)
    vy, vx = cy[valid], cx[valid]
    if len(vy) == 0:
        return False
    return np.any(primary_canvas[vy, vx])


# =========================================================================
# Sampling Helpers
# =========================================================================

def _sample_active_boxes(boxes, rng):
    """Choose which placement boxes will receive a plant."""
    skip_prob = SKIP_CONFIG["skip_probability"]
    min_plants = SKIP_CONFIG["min_plants_per_dish"]
    active = [i for i in range(len(boxes)) if rng.random() >= skip_prob]
    if len(active) < min_plants:
        inactive = [i for i in range(len(boxes)) if i not in active]
        rng.shuffle(inactive)
        while len(active) < min_plants and len(inactive) > 0:
            active.append(inactive.pop())
        active.sort()
    return active


def _sample_size_mix(base_size, num_plants, rng):
    """Assign a size category to each plant, with optional mixing."""
    r = rng.random()
    if r < SIZE_MIX_CONFIG["homogeneous_prob"]:
        return [base_size] * num_plants
    adjacent = SIZE_MIX_CONFIG["adjacency"].get(base_size, [])
    if not adjacent:
        return [base_size] * num_plants
    adj_size = rng.choice(adjacent)
    cumulative = SIZE_MIX_CONFIG["homogeneous_prob"]
    num_different = 1
    for prob_key, nd in [
        ("mix_1_prob", 1), ("mix_2_prob", 2), ("mix_3_prob", 3),
    ]:
        cumulative += SIZE_MIX_CONFIG[prob_key]
        if r < cumulative:
            num_different = min(nd, num_plants - 1)
            break
    sizes = [base_size] * num_plants
    indices = rng.choice(
        num_plants, size=min(num_different, num_plants), replace=False,
    )
    for idx in indices:
        sizes[idx] = adj_size
    return sizes


def _sample_style_for_size(size, homogeneous_style, rng):
    """Pick a primary config for a given size category."""
    styles = STYLE_MIX_CONFIG["styles_by_size"].get(size, [])
    if not styles:
        return None
    if homogeneous_style and homogeneous_style in styles:
        return homogeneous_style
    return rng.choice(styles)


def _get_all_lateral_modes_for_category(category):
    """List all lateral modes available for a category."""
    if category == "short":
        return [
            "A_few_small", "B_few_horizontal", "C_few_arched",
            "D_few_mixed", "E_none",
        ]
    modes = []
    for group in LATERAL_MIX_CONFIG["compatible_groups"].values():
        modes.extend(group)
    return modes


def _sample_lateral_mode(category, homogeneous_mode, rng):
    """Sample a lateral mode, with mixing across compatible groups."""
    all_modes = _get_all_lateral_modes_for_category(category)
    if homogeneous_mode and homogeneous_mode in all_modes:
        return homogeneous_mode
    if rng.random() < 0.6:
        groups = list(LATERAL_MIX_CONFIG["compatible_groups"].values())
        if category == "short":
            groups.append(["E_none"])
        group = groups[rng.integers(0, len(groups))]
        return rng.choice(group)
    return rng.choice(all_modes)


# =========================================================================
# Root Index: Scan Orchestrator Output
# =========================================================================

def build_root_index(rsa_input_dir):
    """Scan the orchestrator output and index all available roots.

    Returns:
        ``{primary_config: {lateral_mode: [list of entry dicts]}}``
        where each entry has ``root_id``, ``mask_path``, ``labels_path``,
        ``json_path``.
    """
    index = defaultdict(lambda: defaultdict(list))

    primary_configs = sorted([
        d for d in os.listdir(rsa_input_dir)
        if os.path.isdir(os.path.join(rsa_input_dir, d))
        and not d.startswith(".")
    ])

    for pc in primary_configs:
        pc_dir = os.path.join(rsa_input_dir, pc)
        lateral_modes = sorted([
            d for d in os.listdir(pc_dir)
            if os.path.isdir(os.path.join(pc_dir, d))
        ])
        for lm in lateral_modes:
            lm_dir = os.path.join(pc_dir, lm)
            mask_files = sorted(
                glob.glob(os.path.join(lm_dir, "*_full_mask.png")),
            )
            for mf in mask_files:
                root_id = os.path.basename(mf).replace("_full_mask.png", "")
                labels_path = mf.replace("_full_mask.png", "_full_labels.png")
                json_path = mf.replace("_full_mask.png", "_full.json")
                if os.path.exists(labels_path):
                    index[pc][lm].append({
                        "root_id": root_id,
                        "mask_path": mf,
                        "labels_path": labels_path,
                        "json_path": json_path,
                    })

    return dict(index)


def _load_root(entry):
    """Load a single root from disk into a placement-ready dict."""
    mask = np.array(Image.open(entry["mask_path"]))
    labels = np.array(Image.open(entry["labels_path"]))

    if np.sum(mask > 0) == 0:
        return None

    ys, xs = np.nonzero(mask)
    top_y_idx = np.argmin(ys)
    plant_top_y = ys[top_y_idx]
    plant_top_x = int(np.mean(xs[ys == ys[top_y_idx]]))

    primary_mask = labels == 1
    primary_ys, primary_xs = np.nonzero(primary_mask)

    lateral_mode = "unknown"
    if os.path.exists(entry["json_path"]):
        try:
            with open(entry["json_path"], "r") as f:
                meta = json.load(f)
            lateral_mode = meta.get("lateral_mode", "unknown")
        except Exception:
            pass

    return {
        "root_id": entry["root_id"],
        "mask": mask,
        "labels": labels,
        "ys": ys,
        "xs": xs,
        "plant_top_y": plant_top_y,
        "plant_top_x": plant_top_x,
        "primary_ys": primary_ys,
        "primary_xs": primary_xs,
        "lateral_mode": lateral_mode,
    }


# =========================================================================
# Dish Composition
# =========================================================================

def compose_single_dish(root_index, root_pool, base_size, base_style, seed):
    """Compose one petri-dish image from a pool of pre-generated roots.

    Places plants into placement boxes on a shared canvas, handling
    collision avoidance, boundary correction, overlap tracking, and
    label encoding.

    Args:
        root_index: full index dict (for fallback lookups).
        root_pool:  mutable ``{config: {mode: [entries]}}`` — entries
                    are popped as they are consumed.
        base_size:  base size category for this dish.
        base_style: base primary config for this dish.
        seed:       random seed.

    Returns:
        Result dict with ``canvas_mask``, ``canvas_labels``,
        ``overlap_labels``, ``metadata``, etc., or ``None`` if fewer
        than the minimum number of plants could be placed.
    """
    rng = np.random.default_rng(seed)

    canvas_w = CANVAS_CONFIG["width"]
    canvas_h = CANVAS_CONFIG["height"]
    canvas_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    canvas_labels = np.zeros((canvas_h, canvas_w), dtype=np.uint16)
    overlap_labels = np.zeros((canvas_h, canvas_w), dtype=np.uint16)
    primary_canvas = np.zeros((canvas_h, canvas_w), dtype=bool)

    boxes = _calculate_boxes()
    active_boxes = _sample_active_boxes(boxes, rng)
    num_plants = len(active_boxes)
    sizes = _sample_size_mix(base_size, num_plants, rng)

    style_homo = rng.random() < STYLE_MIX_CONFIG["homogeneous_prob"]
    homo_style = base_style if style_homo else None

    lat_homo = rng.random() < LATERAL_MIX_CONFIG["homogeneous_prob"]
    homo_lat_mode = (
        _sample_lateral_mode(base_size, None, rng) if lat_homo else None
    )

    plants_metadata = []
    plants_placed = 0
    roots_used = []

    for plant_idx, box_idx in enumerate(active_boxes):
        plant_id = plant_idx + 1
        box = boxes[box_idx]
        size = sizes[plant_idx]

        style = _sample_style_for_size(
            size, homo_style if size == base_size else None, rng,
        )
        lat_mode = _sample_lateral_mode(
            size, homo_lat_mode if size == base_size else None, rng,
        )

        # --- Find a root to place ---
        root_data = _pop_root(root_pool, style, lat_mode, rng)
        if root_data is None:
            root_data = _pop_root(root_pool, base_style, None, rng)
        if root_data is None:
            root_data = _pop_any_root(root_pool, rng)
        if root_data is None:
            continue

        actual_lat_mode = root_data["lateral_mode"]
        prim_ys = root_data["primary_ys"]
        prim_xs = root_data["primary_xs"]
        ys = root_data["ys"]
        xs = root_data["xs"]
        labels = root_data["labels"]
        top_y = root_data["plant_top_y"]
        top_x = root_data["plant_top_x"]

        # --- Placement with collision avoidance ---
        placed = False
        final_ox, final_oy = 0, 0

        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            px, py = _sample_placement(box, rng)
            ox = int(px - top_x)
            oy = int(py - top_y)

            if box_idx in EDGE_BOXES and len(prim_xs) > 0:
                ox += _compute_boundary_correction(
                    prim_xs, ox, canvas_w, BOUNDARY_MARGIN,
                )

            if len(prim_ys) > 0:
                if _check_primary_collision(
                    primary_canvas, prim_ys, prim_xs, oy, ox,
                    canvas_h, canvas_w,
                ):
                    continue

            placed = True
            final_ox, final_oy = ox, oy
            break

        if not placed:
            continue

        # --- Composite onto canvas ---
        cy = ys + final_oy
        cx = xs + final_ox
        lv = labels[ys, xs]

        valid = (
            (cy >= 0) & (cy < canvas_h) & (cx >= 0)
            & (cx < canvas_w) & (lv > 0)
        )
        vy, vx = cy[valid], cx[valid]
        vl = lv[valid]

        is_primary = vl == 1
        is_lateral = vl > 1

        new_labels = np.zeros(len(vl), dtype=np.uint16)
        new_labels[is_primary] = encode_label(plant_id, None)
        if np.any(is_lateral):
            lateral_ids = vl[is_lateral] - 1
            new_labels[is_lateral] = (
                plant_id * 100 + lateral_ids
            ).astype(np.uint16)

        canvas_mask[vy, vx] = 255
        primary_canvas[vy[is_primary], vx[is_primary]] = True

        # Overlap tracking
        existing = canvas_labels[vy, vx]
        has_existing = existing > 0
        overlap_empty = overlap_labels[vy, vx] == 0
        overlap_idx = has_existing & overlap_empty
        overlap_labels[vy[overlap_idx], vx[overlap_idx]] = (
            new_labels[overlap_idx]
        )
        no_existing = ~has_existing
        canvas_labels[vy[no_existing], vx[no_existing]] = (
            new_labels[no_existing]
        )

        plants_placed += 1
        roots_used.append(root_data["root_id"])
        plants_metadata.append({
            "plant_id": plant_id,
            "box_id": box_idx,
            "source_file": root_data["root_id"],
            "primary_config": style,
            "lateral_mode": actual_lat_mode,
            "size_category": size,
        })

    if plants_placed < SKIP_CONFIG["min_plants_per_dish"]:
        return None

    # Overlap summary
    overlap_coords = np.where(overlap_labels > 0)
    overlap_summary = {}
    for y, x in zip(overlap_coords[0], overlap_coords[1]):
        key = tuple(
            sorted([int(canvas_labels[y, x]), int(overlap_labels[y, x])]),
        )
        if key not in overlap_summary:
            overlap_summary[key] = {"labels": list(key), "pixel_count": 0}
        overlap_summary[key]["pixel_count"] += 1

    # Dominant primary config
    config_counts = {}
    for p in plants_metadata:
        c = p["primary_config"]
        config_counts[c] = config_counts.get(c, 0) + 1

    dominant_config = base_style
    if config_counts:
        max_count = max(config_counts.values())
        tied = [k for k, v in config_counts.items() if v == max_count]
        dominant_config = base_style if base_style in tied else tied[0]

    return {
        "canvas_mask": canvas_mask,
        "canvas_labels": canvas_labels,
        "overlap_labels": overlap_labels,
        "roots_used": roots_used,
        "plants_placed": plants_placed,
        "metadata": {
            "canvas_size": [canvas_h, canvas_w],
            "base_size": base_size,
            "base_style": base_style,
            "dominant_primary_config": dominant_config,
            "active_boxes": active_boxes,
            "plants": plants_metadata,
            "overlap_summary": list(overlap_summary.values()),
            "total_overlap_pixels": int(len(overlap_coords[0])),
            "seed": seed,
        },
    }


# =========================================================================
# Root Pool Helpers (Internal)
# =========================================================================

def _pop_root(root_pool, config, mode, rng):
    """Pop a loaded root from the pool, or ``None`` if unavailable."""
    if config not in root_pool:
        return None

    if mode is not None and mode in root_pool[config]:
        entries = root_pool[config][mode]
        if entries:
            return _load_root(entries.pop())

    modes = list(root_pool[config].keys())
    if modes:
        rng.shuffle(modes)
        for m in modes:
            if root_pool[config][m]:
                return _load_root(root_pool[config][m].pop())
    return None


def _pop_any_root(root_pool, rng):
    """Pop any available root from any config/mode."""
    configs = list(root_pool.keys())
    if not configs:
        return None
    rng.shuffle(configs)
    for c in configs:
        result = _pop_root(root_pool, c, None, rng)
        if result is not None:
            return result
    return None


def _count_pool(root_pool):
    """Count total remaining roots in the pool."""
    return sum(
        len(entries)
        for modes in root_pool.values()
        for entries in modes.values()
    )


def _build_pool_subset(root_index, rng, roots_per_mode=3):
    """Build a small random pool subset for a multiprocessing worker."""
    pool = defaultdict(lambda: defaultdict(list))
    for config in root_index:
        for mode in root_index[config]:
            entries = root_index[config][mode]
            if len(entries) <= roots_per_mode:
                pool[config][mode] = list(entries)
            else:
                indices = rng.choice(
                    len(entries), size=roots_per_mode, replace=False,
                )
                pool[config][mode] = [entries[i] for i in indices]
    return dict(pool)


# =========================================================================
# Disk I/O
# =========================================================================

def _save_dish(result, output_dir, dish_id):
    """Save a composed dish to disk (mask, labels, overlap, metadata)."""
    os.makedirs(output_dir, exist_ok=True)
    Image.fromarray(result["canvas_mask"]).save(
        os.path.join(output_dir, f"{dish_id}_mask.png"),
    )
    Image.fromarray(result["canvas_labels"]).save(
        os.path.join(output_dir, f"{dish_id}_labels.png"),
    )
    Image.fromarray(result["overlap_labels"]).save(
        os.path.join(output_dir, f"{dish_id}_overlap.png"),
    )

    result["metadata"]["dish_id"] = dish_id
    result["metadata"]["generation_timestamp"] = datetime.now().isoformat()
    with open(
        os.path.join(output_dir, f"{dish_id}_metadata.json"), "w",
    ) as f:
        json.dump(result["metadata"], f, indent=2)


# =========================================================================
# Multiprocessing Worker
# =========================================================================

def _worker_generate_dish(task):
    """Worker: generate and save one dish (for multiprocessing)."""
    dish_idx, rsa_input_dir, output_dir, base_seed, base_configs = task
    seed = base_seed + dish_idx * 7 + 13
    rng = np.random.default_rng(seed)

    base_style = rng.choice(base_configs)
    base_size = CONFIG_TO_CATEGORY.get(base_style, "medium")

    root_index = build_root_index(rsa_input_dir)
    root_pool = _build_pool_subset(root_index, rng, roots_per_mode=3)

    result = compose_single_dish(
        root_index, root_pool, base_size, base_style, seed,
    )

    if result is None:
        return {"status": "failed", "dish_idx": dish_idx}

    dish_id = f"dish_{dish_idx:06d}"
    _save_dish(result, output_dir, dish_id)

    return {
        "status": "ok",
        "dish_idx": dish_idx,
        "dish_id": dish_id,
        "plants_placed": result["plants_placed"],
        "total_overlap_pixels": result["metadata"]["total_overlap_pixels"],
        "base_style": base_style,
        "dominant_primary_config": result["metadata"]["dominant_primary_config"],
    }


# =========================================================================
# Sequential Generation
# =========================================================================

def _generate_sequential(rsa_input_dir, output_dir, num_dishes,
                         base_seed, verbose):
    """Generate dishes sequentially with a shared root pool."""
    root_index = build_root_index(rsa_input_dir)
    base_configs = sorted(root_index.keys())

    if not base_configs:
        print("ERROR: No root configs found in input directory.")
        return []

    if verbose:
        total_roots = sum(
            len(entries)
            for modes in root_index.values()
            for entries in modes.values()
        )
        print(f"Indexed {total_roots} roots across {len(base_configs)} configs")

    root_pool = {
        config: {mode: list(root_index[config][mode]) for mode in root_index[config]}
        for config in root_index
    }

    rng = np.random.default_rng(base_seed)
    results = []
    generated = 0
    attempts = 0
    max_attempts = num_dishes * 3

    pbar = tqdm(
        total=num_dishes, desc="Generating dishes", disable=not verbose,
    )

    while generated < num_dishes and attempts < max_attempts:
        attempts += 1
        seed = base_seed + attempts * 7

        base_style = rng.choice(base_configs)
        base_size = CONFIG_TO_CATEGORY.get(base_style, "medium")

        dish = compose_single_dish(
            root_index, root_pool, base_size, base_style, seed,
        )

        if dish is None:
            continue

        dish_id = f"dish_{generated:06d}"
        _save_dish(dish, output_dir, dish_id)

        results.append({
            "status": "ok",
            "dish_idx": generated,
            "dish_id": dish_id,
            "plants_placed": dish["plants_placed"],
            "total_overlap_pixels": dish["metadata"]["total_overlap_pixels"],
            "base_style": base_style,
            "dominant_primary_config": dish["metadata"]["dominant_primary_config"],
        })
        generated += 1
        pbar.update(1)

        remaining = _count_pool(root_pool)
        if remaining < SKIP_CONFIG["min_plants_per_dish"]:
            if verbose:
                print(f"\nRoot pool exhausted after {generated} dishes.")
            break

    pbar.close()
    return results


# =========================================================================
# Main Entry Point
# =========================================================================

def generate_all_dishes(
    rsa_input_dir,
    output_dir="petri_dishes",
    num_dishes=1000,
    base_seed=42,
    clear_existing=True,
    num_workers=0,
    verbose=True,
):
    """Generate petri-dish compositions from RSA orchestrator output.

    Args:
        rsa_input_dir:  path to RSA orchestrator output folder.
        output_dir:     destination for dish files.
        num_dishes:     total number of dishes to generate.
        base_seed:      random seed.
        clear_existing: if ``True``, remove *output_dir* first.
        num_workers:    ``0`` = sequential, ``-1`` = all cores, ``N`` = N cores.
        verbose:        print progress.

    Returns:
        Summary dict with generation statistics.
    """
    if clear_existing and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if num_workers == -1:
        num_workers = mp.cpu_count()
    elif num_workers > 0:
        num_workers = min(num_workers, mp.cpu_count())

    mode_str = f"{num_workers} workers" if num_workers > 0 else "sequential"
    start_time = datetime.now()

    if verbose:
        print(f"{'=' * 60}")
        print("pyrootsim — PETRI DISH GENERATION")
        print(f"{'=' * 60}")
        print(f"Input:       {rsa_input_dir}")
        print(f"Output:      {output_dir}")
        print(f"Dishes:      {num_dishes}")
        print(f"Execution:   {mode_str}")
        print(f"Min plants:  {SKIP_CONFIG['min_plants_per_dish']}")
        print(f"{'=' * 60}")

    if num_workers > 0:
        root_index = build_root_index(rsa_input_dir)
        base_configs = sorted(root_index.keys())

        if verbose:
            total_roots = sum(
                len(e) for m in root_index.values() for e in m.values()
            )
            print(
                f"Indexed {total_roots} roots"
                f" across {len(base_configs)} configs"
            )

        tasks = [
            (i, rsa_input_dir, output_dir, base_seed, base_configs)
            for i in range(num_dishes)
        ]

        results = []
        with mp.Pool(processes=num_workers) as pool:
            for r in tqdm(
                pool.imap_unordered(_worker_generate_dish, tasks),
                total=num_dishes,
                desc="Generating dishes",
                disable=not verbose,
                smoothing=0.02,
            ):
                results.append(r)
    else:
        results = _generate_sequential(
            rsa_input_dir, output_dir, num_dishes, base_seed, verbose,
        )

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    # Build summary
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "failed"]

    plants_dist = defaultdict(int)
    for r in ok:
        plants_dist[r["plants_placed"]] += 1

    summary = {
        "generator_version": "pyrootsim",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "dishes_per_second": round(len(ok) / max(elapsed, 0.1), 2),
        "rsa_input_dir": rsa_input_dir,
        "output_dir": output_dir,
        "num_requested": num_dishes,
        "num_generated": len(ok),
        "num_failed": len(failed),
        "num_workers": num_workers,
        "base_seed": base_seed,
        "min_plants_per_dish": SKIP_CONFIG["min_plants_per_dish"],
        "plants_per_dish_distribution": dict(plants_dist),
        "avg_plants_per_dish": (
            sum(k * v for k, v in plants_dist.items()) / max(len(ok), 1)
        ),
        "avg_overlap_pixels": (
            sum(r["total_overlap_pixels"] for r in ok) / max(len(ok), 1)
        ),
    }

    summary_path = os.path.join(output_dir, "generation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # CSV for downstream stratified splitting
    csv_path = os.path.join(output_dir, "dish_metadata.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dish_id", "base_style", "dominant_primary_config",
            "plant_count", "overlap_pixels",
        ])
        for r in ok:
            writer.writerow([
                r["dish_id"], r["base_style"],
                r["dominant_primary_config"],
                r["plants_placed"], r["total_overlap_pixels"],
            ])

    if verbose:
        _print_report(summary)

    return summary


def _print_report(summary):
    """Print a human-readable generation report."""
    print(f"\n{'=' * 60}")
    print("GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(
        f"Time:         {summary['elapsed_seconds']:.1f}s"
        f" ({summary['dishes_per_second']:.1f} dishes/sec)"
    )
    print(
        f"Generated:    {summary['num_generated']}"
        f" / {summary['num_requested']}"
    )
    print(f"Failed:       {summary['num_failed']}")
    print(f"Avg plants:   {summary['avg_plants_per_dish']:.2f}")
    print(f"Avg overlap:  {summary['avg_overlap_pixels']:.0f} px")

    dist = summary["plants_per_dish_distribution"]
    total = summary["num_generated"]
    print("\nPlants per dish distribution:")
    for count in sorted(dist.keys(), key=int):
        freq = dist[count]
        pct = freq / total * 100 if total > 0 else 0
        bar = "█" * max(1, freq * 40 // max(total, 1))
        print(f"  {count} plants: {freq:5d} ({pct:5.1f}%) {bar}")

    print(f"\nSummary: {summary['output_dir']}/generation_summary.json")
    print(f"CSV Metadata: {summary['output_dir']}/dish_metadata.csv")


# =========================================================================
# CLI Entry Point
# =========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="pyrootsim — Petri Dish Composer",
    )
    parser.add_argument(
        "rsa_input_dir", type=str,
        help="Path to RSA orchestrator output folder",
    )
    parser.add_argument("--output-dir", type=str, default="petri_dishes")
    parser.add_argument("--num-dishes", type=int, default=1000)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="0=sequential, -1=all cores, N=use N cores",
    )
    parser.add_argument("--no-clear", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    generate_all_dishes(
        rsa_input_dir=args.rsa_input_dir,
        output_dir=args.output_dir,
        num_dishes=args.num_dishes,
        base_seed=args.base_seed,
        clear_existing=not args.no_clear,
        num_workers=args.num_workers,
        verbose=not args.quiet,
    )