"""
Petri dish visualiser for PyRootsim.

Colour-coded rendering of composed petri-dish images for visual
inspection.  Two rendering modes are available:

* **root_type** — Primary roots are white, each lateral gets a unique
  colour (cycled per plant), and overlap pixels are blended.
* **config** — Each plant is coloured by its primary config, with
  laterals drawn in a lighter shade and overlaps blended.

Usage::

    from pyrootsim.dish.visualizer import visualize_dish, visualize_dishes_grid

    # Single dish
    fig = visualize_dish("petri_dishes/dish_000000")

    # Grid of dishes
    fig = visualize_dishes_grid("petri_dishes", n=6)

    # Side-by-side dual view
    fig = visualize_dish_dual("petri_dishes/dish_000000")
"""

import os
import json
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


# =========================================================================
# Colour Palettes
# =========================================================================

LATERAL_COLORS = np.array([
    [255,  50,  50], [ 50, 200,  50], [ 50, 100, 255], [255, 220,  40],
    [255,  50, 255], [ 50, 255, 255], [255, 140,  30], [160,  50, 255],
    [ 50, 255, 140], [255,  50, 140], [140, 255,  50], [ 50, 150, 255],
    [255, 160, 140], [140, 255, 160], [140, 140, 255], [255, 210, 120],
], dtype=np.uint8)
"""Per-plant lateral colours (cycled)."""

CONFIG_COLORS = {
    "01_Short_Kinky_Noisy":        [255, 100, 100],
    "02_Short_Smooth_Clean":       [255, 160, 100],
    "03_Short_Kinky_Smooth":       [255, 200, 100],
    "04_Medium_Kinky_Noisy":       [100, 255, 100],
    "05_Medium_Smooth_Snake":      [100, 255, 180],
    "06_Medium_Clean_GroundTruth": [100, 255, 255],
    "07_Long_Kinky_Noisy":         [100, 100, 255],
    "08_Long_Sweeping_Curves":     [160, 100, 255],
    "09_Long_Smooth_Static":       [200, 100, 255],
    "10_ExtraLong_Hybrid":         [255, 100, 255],
    "11_ExtraLong_Curvy_Clean":    [255, 100, 200],
    "12_ExtraLong_Mixed_Sweeping": [200, 200, 100],
}
"""Per-config colours for config-level rendering."""


# =========================================================================
# Label Decoding
# =========================================================================

def decode_label(label_value):
    """Decode a uint16 label into plant ID and root type.

    Returns ``None`` for background (0), otherwise a dict with
    ``plant_id``, ``root_type`` (``"primary"`` or ``"lateral"``),
    and ``lateral_id``.
    """
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


def _get_lateral_color(plant_id, lateral_id):
    """Return a unique colour for a specific lateral on a specific plant."""
    idx = ((plant_id - 1) * 5 + lateral_id) % len(LATERAL_COLORS)
    return LATERAL_COLORS[idx].astype(float)


def _color_for_label(lv):
    """Return the RGB colour for a single label value."""
    decoded = decode_label(lv)
    if decoded is None:
        return np.array([0, 0, 0], dtype=float)
    if decoded["root_type"] == "primary":
        return np.array([255, 255, 255], dtype=float)
    return _get_lateral_color(decoded["plant_id"], decoded["lateral_id"]).astype(float)


# =========================================================================
# Rendering Modes
# =========================================================================

def render_by_root_type(canvas_labels, overlap_labels, metadata=None):
    """Render with primary roots white and laterals in unique colours.

    Overlap pixels are blended as the average of the two overlapping
    root colours.
    """
    h, w = canvas_labels.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for lv in np.unique(canvas_labels):
        if lv == 0:
            continue
        mask = canvas_labels == lv
        decoded = decode_label(int(lv))
        if decoded is None:
            continue
        if decoded["root_type"] == "primary":
            rgb[mask] = [255, 255, 255]
        else:
            color = _get_lateral_color(decoded["plant_id"], decoded["lateral_id"])
            rgb[mask] = color.astype(np.uint8)

    overlap_mask = overlap_labels > 0
    if np.any(overlap_mask):
        oy, ox = np.where(overlap_mask)
        for y, x in zip(oy, ox):
            c1 = _color_for_label(int(canvas_labels[y, x]))
            c2 = _color_for_label(int(overlap_labels[y, x]))
            rgb[y, x] = ((c1 + c2) / 2).astype(np.uint8)

    return rgb


def render_by_config(canvas_labels, overlap_labels, metadata):
    """Render with colours based on each plant's primary config.

    Primary roots use the full config colour; laterals use a lighter
    shade. Overlap pixels are blended.
    """
    h, w = canvas_labels.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    plant_config = {}
    if metadata and "plants" in metadata:
        for p in metadata["plants"]:
            plant_config[p["plant_id"]] = p.get("primary_config", "unknown")

    for lv in np.unique(canvas_labels):
        if lv == 0:
            continue
        mask = canvas_labels == lv
        decoded = decode_label(int(lv))
        if decoded is None:
            continue

        config = plant_config.get(decoded["plant_id"], "unknown")
        base_color = np.array(
            CONFIG_COLORS.get(config, [180, 180, 180]), dtype=float,
        )

        if decoded["root_type"] == "primary":
            rgb[mask] = np.clip(base_color, 0, 255).astype(np.uint8)
        else:
            lighter = base_color * 0.6 + 100
            rgb[mask] = np.clip(lighter, 0, 255).astype(np.uint8)

    overlap_mask = overlap_labels > 0
    if np.any(overlap_mask):
        oy, ox = np.where(overlap_mask)
        for y, x in zip(oy, ox):
            d1 = decode_label(int(canvas_labels[y, x]))
            d2 = decode_label(int(overlap_labels[y, x]))
            cfg1 = plant_config.get(d1["plant_id"], "unknown") if d1 else "unknown"
            cfg2 = plant_config.get(d2["plant_id"], "unknown") if d2 else "unknown"
            c1 = np.array(CONFIG_COLORS.get(cfg1, [180, 180, 180]), dtype=float)
            c2 = np.array(CONFIG_COLORS.get(cfg2, [180, 180, 180]), dtype=float)
            rgb[y, x] = ((c1 + c2) / 2).astype(np.uint8)

    return rgb


# =========================================================================
# Disk Loading
# =========================================================================

def load_dish(dish_prefix):
    """Load a dish from disk.

    Args:
        dish_prefix: path prefix without extension, e.g.
                     ``"petri_dishes/dish_000000"``.

    Returns:
        ``(canvas_labels, overlap_labels, mask, metadata)``
    """
    if dish_prefix.endswith("_mask.png"):
        dish_prefix = dish_prefix.replace("_mask.png", "")

    labels_path = f"{dish_prefix}_labels.png"
    overlap_path = f"{dish_prefix}_overlap.png"
    mask_path = f"{dish_prefix}_mask.png"
    json_path = f"{dish_prefix}_metadata.json"

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    canvas_labels = np.array(Image.open(labels_path))
    overlap_labels = (
        np.array(Image.open(overlap_path))
        if os.path.exists(overlap_path)
        else np.zeros_like(canvas_labels)
    )
    mask = (
        np.array(Image.open(mask_path))
        if os.path.exists(mask_path)
        else None
    )

    metadata = None
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            metadata = json.load(f)

    return canvas_labels, overlap_labels, mask, metadata


# =========================================================================
# Visualisation Functions
# =========================================================================

def visualize_dish(dish_prefix, mode="root_type", figsize=(16, 15),
                   title=None, show_legend=True, ax=None):
    """Visualise a single petri dish.

    Args:
        dish_prefix: path prefix (e.g. ``"output/dish_000000"``).
        mode:        ``"root_type"`` or ``"config"``.
        figsize:     figure size (only used if *ax* is ``None``).
        title:       custom title (auto-generated if ``None``).
        show_legend: show colour legend (config mode only).
        ax:          optional matplotlib axes to draw on.

    Returns:
        Figure, or ``None`` if *ax* was provided.
    """
    canvas_labels, overlap_labels, mask, metadata = load_dish(dish_prefix)

    if mode == "config":
        rgb = render_by_config(canvas_labels, overlap_labels, metadata)
    else:
        rgb = render_by_root_type(canvas_labels, overlap_labels, metadata)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = None

    ax.imshow(rgb, interpolation="nearest")
    ax.axis("off")

    if title is None:
        dish_name = os.path.basename(dish_prefix)
        n_plants = metadata.get("plants", []) if metadata else []
        overlap_px = metadata.get("total_overlap_pixels", 0) if metadata else 0
        base = metadata.get("base_style", "?") if metadata else "?"
        title = (
            f"{dish_name}  |  {len(n_plants)} plants"
            f"  |  {overlap_px} overlap px  |  base: {base}"
        )
    ax.set_title(title, fontsize=11)

    if show_legend and metadata and mode == "config":
        configs_used = {
            p.get("primary_config", "unknown")
            for p in metadata.get("plants", [])
        }
        patches = []
        for cfg in sorted(configs_used):
            c = np.array(CONFIG_COLORS.get(cfg, [180, 180, 180])) / 255.0
            patches.append(mpatches.Patch(color=c, label=cfg))
        if patches:
            ax.legend(
                handles=patches, loc="lower left", fontsize=7, framealpha=0.8,
            )

    if own_fig:
        plt.tight_layout()
        return fig
    return None


def visualize_dishes_grid(output_dir, n=6, mode="root_type", cols=3,
                          figsize_per_dish=(8, 7.5), seed=None):
    """Visualise a grid of dishes from an output directory.

    Args:
        output_dir:       petri dish output directory.
        n:                number of dishes to show.
        mode:             ``"root_type"`` or ``"config"``.
        cols:             columns in the grid.
        figsize_per_dish: size per subplot.
        seed:             if set, randomly sample; otherwise take first *n*.

    Returns:
        Figure, or ``None`` if no dishes found.
    """
    mask_files = sorted(glob.glob(os.path.join(output_dir, "*_mask.png")))
    if not mask_files:
        print(f"No dishes found in {output_dir}")
        return None

    if seed is not None:
        rng = np.random.default_rng(seed)
        indices = rng.choice(
            len(mask_files), size=min(n, len(mask_files)), replace=False,
        )
        mask_files = [mask_files[i] for i in sorted(indices)]
    else:
        mask_files = mask_files[:n]

    n_actual = len(mask_files)
    rows = (n_actual + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(figsize_per_dish[0] * cols, figsize_per_dish[1] * rows),
    )
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for i, mf in enumerate(mask_files):
        r, c = i // cols, i % cols
        prefix = mf.replace("_mask.png", "")
        visualize_dish(
            prefix, mode=mode, ax=axes[r, c],
            show_legend=(mode == "config"),
        )

    for i in range(n_actual, rows * cols):
        r, c = i // cols, i % cols
        axes[r, c].axis("off")

    fig.suptitle(
        f"Petri Dishes — {mode} colouring  ({n_actual} dishes)", fontsize=14,
    )
    plt.tight_layout()
    return fig


def visualize_dish_dual(dish_prefix, figsize=(28, 13)):
    """Show both root_type and config views side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    visualize_dish(dish_prefix, mode="root_type", ax=ax1, title="By root type")
    visualize_dish(dish_prefix, mode="config", ax=ax2, title="By primary config")
    plt.tight_layout()
    return fig


# =========================================================================
# Statistics Helper
# =========================================================================

def print_dish_stats(output_dir):
    """Print summary statistics for a dish output directory."""
    json_files = sorted(
        glob.glob(os.path.join(output_dir, "*_metadata.json")),
    )
    if not json_files:
        print(f"No dishes found in {output_dir}")
        return

    n = len(json_files)
    plants_counts = []
    overlap_counts = []
    configs_used = set()
    modes_used = set()

    for jf in json_files:
        with open(jf, "r") as f:
            meta = json.load(f)
        plants_counts.append(len(meta.get("plants", [])))
        overlap_counts.append(meta.get("total_overlap_pixels", 0))
        for p in meta.get("plants", []):
            configs_used.add(p.get("primary_config", "?"))
            modes_used.add(p.get("lateral_mode", "?"))

    plants_counts = np.array(plants_counts)
    overlap_counts = np.array(overlap_counts)

    print(f"Dishes: {n}")
    print(
        f"Plants/dish: mean={plants_counts.mean():.1f},"
        f" min={plants_counts.min()}, max={plants_counts.max()}"
    )
    print(
        f"Overlap px: mean={overlap_counts.mean():.0f},"
        f" max={overlap_counts.max()}"
    )
    print(f"Unique configs: {len(configs_used)}")
    print(f"Unique lateral modes: {len(modes_used)}")

    print("\nPlants distribution:")
    for c in sorted(np.unique(plants_counts)):
        freq = np.sum(plants_counts == c)
        pct = freq / n * 100
        print(f"  {c} plants: {freq} ({pct:.1f}%)")