"""
RSA dataset orchestrator for PyRootSim.

Generates a complete root system architecture dataset across all primary
configurations and all lateral modes.  Supports sequential or
multiprocessing execution.

Output structure::

    output_dir/
        {primary_config}/
            {lateral_mode}/
                {root_id}_full_mask.png
                {root_id}_full_labels.png
                {root_id}_full.json
        generation_summary.json

Usage examples::

    from pyrootsim.dataset.orchestrator import generate_rsa_dataset

    # Sequential
    summary = generate_rsa_dataset(output_dir="RSA_dataset", num_per_cell=10)

    # Multiprocessing (all cores)
    summary = generate_rsa_dataset(output_dir="RSA_dataset", num_per_cell=10,
                                   num_workers=-1)

    # Multiprocessing (4 cores)
    summary = generate_rsa_dataset(output_dir="RSA_dataset", num_per_cell=10,
                                   num_workers=4)

The module can also be run directly from the command line::

    python -m pyrootsim.dataset.orchestrator --output-dir RSA_dataset \\
        --num-per-cell 10 --num-workers -1
"""

import os
import json
import shutil
import multiprocessing as mp
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm

from pyrootsim.roots.configs import PRIMARY_CONFIGS, LATERAL_MODES
from pyrootsim.roots.primary import generate_single_primary_root
from pyrootsim.roots.lateral import generate_lateral_root_inline


# =========================================================================
# Config Resolution
# =========================================================================

# Primary configs to generate (skip sub-configs used only as mix sources)
ALL_PRIMARY_CONFIGS = sorted([
    k for k in PRIMARY_CONFIGS
    if k not in {"12a_ExtraLong_Gentle", "12b_ExtraLong_Sweeping"}
])


def get_category_for_config(config_name):
    """Resolve the lateral-mode category for a primary config.

    Mix-source configs delegate to the category of their first source.
    """
    cfg = PRIMARY_CONFIGS[config_name]
    if "mix_sources" in cfg:
        src = cfg["mix_sources"][0]
        return PRIMARY_CONFIGS[src]["category"]
    return cfg["category"]


def get_lateral_modes_for_config(config_name):
    """Return all lateral mode keys available for *config_name*'s category."""
    category = get_category_for_config(config_name)
    return sorted(LATERAL_MODES[category].keys())


def build_generation_grid(primary_configs=None):
    """Build the full ``(primary_config, lateral_mode, category)`` grid.

    Args:
        primary_configs: list of primary config keys, or ``None`` for all 12.

    Returns:
        List of ``(primary_config, lateral_mode, category)`` tuples.
    """
    if primary_configs is None:
        primary_configs = ALL_PRIMARY_CONFIGS

    grid = []
    for cfg in primary_configs:
        category = get_category_for_config(cfg)
        for mode in get_lateral_modes_for_config(cfg):
            grid.append((cfg, mode, category))
    return grid


# =========================================================================
# Single Root Generation (Unit of Work)
# =========================================================================

def generate_single_root(primary_config, lateral_mode, root_idx, base_seed):
    """Generate one complete root (primary + laterals) entirely in memory.

    A deterministic seed is derived from the config name, lateral mode,
    and root index so that results are reproducible.

    Args:
        primary_config: primary config key.
        lateral_mode:   lateral mode key.
        root_idx:       index within this cell.
        base_seed:      base seed for reproducibility.

    Returns:
        Composite result dict, or ``None`` if generation failed.
    """
    seed = (
        base_seed + root_idx
        + abs(hash((primary_config, lateral_mode))) % 10 ** 7
    )

    primary = generate_single_primary_root(primary_config, root_idx, seed)
    if primary is None:
        return None

    lateral_seed = seed + 500_000
    result = generate_lateral_root_inline(
        primary, primary_config, lateral_mode, lateral_seed,
    )
    if result is None:
        return None

    # Enrich metadata for downstream aggregation
    result["metadata"]["primary_config"] = primary_config
    result["metadata"]["lateral_mode"] = lateral_mode
    result["metadata"]["cell_root_idx"] = root_idx
    result["metadata"]["seed"] = seed
    result["metadata"]["primary_width_info"] = primary["metadata"].get(
        "width_info", {},
    )

    return result


# =========================================================================
# Save Helpers
# =========================================================================

def save_root_result(result, output_dir, primary_config, lateral_mode):
    """Save a single composite root result to disk.

    Writes three files into ``output_dir/primary_config/lateral_mode/``:
    a mask PNG, a label PNG, and a JSON metadata file.
    """
    cell_dir = os.path.join(output_dir, primary_config, lateral_mode)
    os.makedirs(cell_dir, exist_ok=True)

    root_id = result["root_id"]

    Image.fromarray(result["combined_mask"]).save(
        os.path.join(cell_dir, f"{root_id}_full_mask.png"),
    )
    Image.fromarray(result["label_img"]).save(
        os.path.join(cell_dir, f"{root_id}_full_labels.png"),
    )
    with open(os.path.join(cell_dir, f"{root_id}_full.json"), "w") as f:
        json.dump(result["metadata"], f, indent=2)

    return root_id


# =========================================================================
# Multiprocessing Worker
# =========================================================================

def _worker_generate_and_save(task):
    """Generate one root and save it to disk (multiprocessing worker).

    Args:
        task: ``(primary_config, lateral_mode, root_idx, base_seed, output_dir)``

    Returns:
        Status dict for aggregation.
    """
    primary_config, lateral_mode, root_idx, base_seed, output_dir = task

    try:
        result = generate_single_root(
            primary_config, lateral_mode, root_idx, base_seed,
        )

        if result is None:
            return {
                "status": "failed",
                "primary_config": primary_config,
                "lateral_mode": lateral_mode,
                "root_idx": root_idx,
            }

        root_id = save_root_result(
            result, output_dir, primary_config, lateral_mode,
        )

        pri_wi = result["metadata"].get("primary_width_info", {})
        pri_pinch_n = pri_wi.get("num_pinch_events", 0)

        lat_pinch_n = sum(
            lat.get("num_pinch_events", 0)
            for lat in result["metadata"].get("laterals", [])
        )

        return {
            "status": "ok",
            "primary_config": primary_config,
            "lateral_mode": lateral_mode,
            "root_idx": root_idx,
            "root_id": root_id,
            "num_laterals": result["metadata"].get("num_laterals", 0),
            "num_top_tip": result["metadata"].get("num_top_tip_laterals", 0),
            "cc8_retries": result["metadata"].get("composite_cc8_retries", 0),
            "cc8_fallback": result["metadata"].get(
                "composite_cc8_fallback", False,
            ),
            "primary_pinch_events": pri_pinch_n,
            "lateral_pinch_events": lat_pinch_n,
            "primary_is_thin": pri_wi.get("is_thin", False),
            "primary_taper": pri_wi.get("taper_bucket", "unknown"),
        }

    except Exception as e:
        return {
            "status": "error",
            "primary_config": primary_config,
            "lateral_mode": lateral_mode,
            "root_idx": root_idx,
            "error": str(e),
        }


# =========================================================================
# Main Entry Point
# =========================================================================

def generate_rsa_dataset(
    output_dir="RSA_dataset",
    num_per_cell=10,
    primary_configs=None,
    base_seed=42,
    clear_existing=True,
    num_workers=0,
    max_retries_per_root=3,
):
    """Generate a complete RSA dataset across all config × lateral-mode cells.

    Args:
        output_dir:           root output directory.
        num_per_cell:         roots per ``(primary_config, lateral_mode)`` cell.
        primary_configs:      list of primary config keys, or ``None`` for all.
        base_seed:            base random seed.
        clear_existing:       if ``True``, remove *output_dir* first.
        num_workers:          ``0`` = sequential, ``-1`` = all cores,
                              ``N`` = use N cores.
        max_retries_per_root: (reserved) max attempts per root before skip.

    Returns:
        Summary dict with generation statistics.
    """
    if clear_existing and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --- Build grid ---
    grid = build_generation_grid(primary_configs)
    total_cells = len(grid)
    total_roots = total_cells * num_per_cell

    # --- Resolve worker count ---
    if num_workers == -1:
        num_workers = mp.cpu_count()
    elif num_workers > 0:
        num_workers = min(num_workers, mp.cpu_count())

    mode_str = f"{num_workers} workers" if num_workers > 0 else "sequential"

    print(f"{'=' * 70}")
    print("pyrootsim — RSA DATASET GENERATION")
    print(f"{'=' * 70}")
    print(f"Output:          {output_dir}")
    print(f"Primary configs: {len(set(c for c, _, _ in grid))}")
    print(f"Total cells:     {total_cells} (config × lateral_mode)")
    print(f"Roots per cell:  {num_per_cell}")
    print(f"Total roots:     {total_roots}")
    print(f"Execution:       {mode_str}")
    print(f"Base seed:       {base_seed}")
    print(f"{'=' * 70}")

    configs_in_grid = sorted(set(c for c, _, _ in grid))
    for cfg in configs_in_grid:
        modes = [m for c, m, _ in grid if c == cfg]
        print(
            f"  {cfg}: {len(modes)} lateral modes × {num_per_cell}"
            f" = {len(modes) * num_per_cell} roots"
        )
    print()

    # --- Build task list ---
    tasks = [
        (pc, lm, ri, base_seed, output_dir)
        for pc, lm, _cat in grid
        for ri in range(num_per_cell)
    ]

    # --- Execute ---
    results = []
    start_time = datetime.now()

    if num_workers > 0:
        print(f"Starting multiprocessing pool with {num_workers} workers...")
        with mp.Pool(processes=num_workers) as pool:
            for r in tqdm(
                pool.imap_unordered(_worker_generate_and_save, tasks),
                total=len(tasks),
                desc="Generating roots",
                smoothing=0.02,
            ):
                results.append(r)
    else:
        for task in tqdm(tasks, desc="Generating roots"):
            results.append(_worker_generate_and_save(task))

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    # --- Aggregate & save ---
    summary = _build_summary(
        results, grid, num_per_cell, base_seed, output_dir,
        num_workers, start_time, end_time, elapsed,
    )

    summary_path = os.path.join(output_dir, "generation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    _print_report(summary)
    return summary


# =========================================================================
# Summary & Reporting (Internal)
# =========================================================================

def _build_summary(results, grid, num_per_cell, base_seed, output_dir,
                   num_workers, start_time, end_time, elapsed):
    """Aggregate per-root results into a structured summary dict."""
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "failed"]
    errors = [r for r in results if r["status"] == "error"]

    # Per-cell stats
    cell_stats = {}
    for r in results:
        key = (r["primary_config"], r["lateral_mode"])
        if key not in cell_stats:
            cell_stats[key] = {
                "ok": 0, "failed": 0, "error": 0,
                "top_tip_total": 0, "cc8_retries": 0, "cc8_fallbacks": 0,
                "primary_pinch_events": 0, "lateral_pinch_events": 0,
                "thin_count": 0,
            }
        cell_stats[key][r["status"]] += 1
        if r["status"] == "ok":
            cell_stats[key]["top_tip_total"] += r.get("num_top_tip", 0)
            cell_stats[key]["cc8_retries"] += r.get("cc8_retries", 0)
            if r.get("cc8_fallback", False):
                cell_stats[key]["cc8_fallbacks"] += 1
            cell_stats[key]["primary_pinch_events"] += r.get(
                "primary_pinch_events", 0,
            )
            cell_stats[key]["lateral_pinch_events"] += r.get(
                "lateral_pinch_events", 0,
            )
            if r.get("primary_is_thin", False):
                cell_stats[key]["thin_count"] += 1

    # Per-config stats
    config_stats = {}
    for (cfg, mode), stats in cell_stats.items():
        if cfg not in config_stats:
            config_stats[cfg] = {
                "ok": 0, "failed": 0, "error": 0, "modes": {},
                "top_tip_total": 0, "primary_pinch_events": 0,
                "lateral_pinch_events": 0, "thin_count": 0,
            }
        for k in ("ok", "failed", "error", "top_tip_total",
                   "primary_pinch_events", "lateral_pinch_events",
                   "thin_count"):
            config_stats[cfg][k] += stats[k]
        config_stats[cfg]["modes"][mode] = stats

    # Global aggregate stats
    total_pri_pinch = sum(r.get("primary_pinch_events", 0) for r in ok)
    total_lat_pinch = sum(r.get("lateral_pinch_events", 0) for r in ok)
    roots_with_pri_pinch = sum(
        1 for r in ok if r.get("primary_pinch_events", 0) > 0
    )
    total_thin = sum(1 for r in ok if r.get("primary_is_thin", False))

    taper_dist = {"none": 0, "mild": 0, "strong": 0, "unknown": 0}
    for r in ok:
        tb = r.get("primary_taper", "unknown")
        taper_dist[tb] = taper_dist.get(tb, 0) + 1

    return {
        "generator_version": "pyrootsim",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "roots_per_second": round(len(ok) / max(elapsed, 0.1), 1),
        "output_dir": output_dir,
        "base_seed": base_seed,
        "num_per_cell": num_per_cell,
        "num_workers": num_workers,
        "grid_size": len(grid),
        "total_tasks": len(results),
        "total_ok": len(ok),
        "total_failed": len(failed),
        "total_errors": len(errors),
        "top_tip_laterals_total": sum(r.get("num_top_tip", 0) for r in ok),
        "cc8_fallback_total": sum(
            1 for r in ok if r.get("cc8_fallback", False)
        ),
        "pinch_stats": {
            "roots_with_primary_pinches": roots_with_pri_pinch,
            "total_primary_pinch_events": total_pri_pinch,
            "total_lateral_pinch_events": total_lat_pinch,
        },
        "width_stats": {
            "thin_primaries": total_thin,
            "taper_distribution": taper_dist,
        },
        "configs": config_stats,
        "errors_detail": errors[:20] if errors else [],
    }


def _print_report(summary):
    """Print a human-readable generation report to stdout."""
    print(f"\n{'=' * 70}")
    print("GENERATION COMPLETE")
    print(f"{'=' * 70}")
    print(
        f"Time:        {summary['elapsed_seconds']:.1f}s "
        f"({summary['roots_per_second']:.1f} roots/sec)"
    )
    print(f"Generated:   {summary['total_ok']} / {summary['total_tasks']}")
    print(f"Failed:      {summary['total_failed']}")
    print(f"Errors:      {summary['total_errors']}")
    print(
        f"Top-tip lat: {summary['top_tip_laterals_total']}"
        " total across all roots"
    )
    print(f"CC8 fallbk:  {summary['cc8_fallback_total']}")

    ps = summary["pinch_stats"]
    ws = summary["width_stats"]
    print(
        f"Pinches:     {ps['roots_with_primary_pinches']} roots w/ primary"
        f" pinches, {ps['total_primary_pinch_events']} pri events,"
        f" {ps['total_lateral_pinch_events']} lat events"
    )
    print(f"Thin roots:  {ws['thin_primaries']} / {summary['total_ok']}")
    print(f"Taper dist:  {ws['taper_distribution']}")
    print()

    header = (
        f"{'Config':<35} {'OK':>5} {'Fail':>5} {'Err':>4} {'TpTip':>5}"
        f" {'Thin':>5} {'PriPn':>5} {'LatPn':>5} {'Modes':>5}"
    )
    print(header)
    print(
        f"{'-' * 35} {'-' * 5} {'-' * 5} {'-' * 4} {'-' * 5}"
        f" {'-' * 5} {'-' * 5} {'-' * 5} {'-' * 5}"
    )
    for cfg in sorted(summary["configs"]):
        s = summary["configs"][cfg]
        n_modes = len(s["modes"])
        print(
            f"{cfg:<35} {s['ok']:>5} {s['failed']:>5} {s['error']:>4}"
            f" {s['top_tip_total']:>5} {s['thin_count']:>5}"
            f" {s['primary_pinch_events']:>5} {s['lateral_pinch_events']:>5}"
            f" {n_modes:>5}"
        )

    print(
        f"\nSummary saved to: {summary['output_dir']}/generation_summary.json"
    )


# =========================================================================
# Convenience: Print the Generation Grid
# =========================================================================

def print_generation_grid(primary_configs=None):
    """Print the full config × mode generation grid for inspection."""
    grid = build_generation_grid(primary_configs)
    configs = sorted(set(c for c, _, _ in grid))

    print(
        f"Generation grid: {len(grid)} cells"
        f" across {len(configs)} primary configs\n"
    )
    for cfg in configs:
        cat = get_category_for_config(cfg)
        modes = [m for c, m, _ in grid if c == cfg]
        print(f"  {cfg}  (category={cat}, {len(modes)} modes)")
        for m in modes:
            print(f"    - {m}")
    print(f"\nTotal cells: {len(grid)}")


# =========================================================================
# CLI Entry Point
# =========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="pyrootsim — RSA Dataset Generator",
    )
    parser.add_argument(
        "--output-dir", type=str, default="RSA_dataset",
        help="Output directory",
    )
    parser.add_argument(
        "--num-per-cell", type=int, default=10,
        help="Number of roots per (config, mode) cell",
    )
    parser.add_argument(
        "--base-seed", type=int, default=42,
        help="Base random seed",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="0=sequential, -1=all cores, N=use N cores",
    )
    parser.add_argument(
        "--no-clear", action="store_true",
        help="Do not clear output directory before starting",
    )
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="Specific primary configs to generate (default: all 12)",
    )
    parser.add_argument(
        "--print-grid", action="store_true",
        help="Print generation grid and exit",
    )

    args = parser.parse_args()

    if args.print_grid:
        print_generation_grid(args.configs)
    else:
        generate_rsa_dataset(
            output_dir=args.output_dir,
            num_per_cell=args.num_per_cell,
            primary_configs=args.configs,
            base_seed=args.base_seed,
            clear_existing=not args.no_clear,
            num_workers=args.num_workers,
        )