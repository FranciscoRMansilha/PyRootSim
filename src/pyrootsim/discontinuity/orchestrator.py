"""
Occlusion orchestrator for PyRootSim.

Applies 2cc, 3cc, and 4cc discontinuities to composed petri dish images
in sequence, sharing a single :class:`GlobalOcclusionTracker` and mask
across all stages. Supports multiprocessing, density-tier sampling,
post-hoc OBB validation, and detailed reporting.

Pipeline per dish
-----------------
1. Build root data structures (``build_roots_fast``).
2. Sample a density tier → per-CC-type target counts.
3. Place 2cc → 3cc → 4cc occlusions on a shared mask.
4. Post-hoc surgical filter: heal phantom 2cc cuts.
5. Post-hoc OBB validation for all CC types.
6. Save occluded mask, OBB annotations, and metadata.
7. Copy original dish files alongside for a self-contained output.

Input
-----
Petri dish folder produced by :mod:`pyrootsim.dish.composer` (or a
stratified "to-occlude" subset produced by
:func:`pyrootsim.dataset.splitter.split_clean_vs_occluded`).

Output per dish
---------------
- ``{dish_id}_mask.png`` — original mask (copied)
- ``{dish_id}_labels.png`` — original labels (copied)
- ``{dish_id}_overlap.png`` — original overlap (copied, if present)
- ``{dish_id}_metadata.json`` — original metadata (copied)
- ``{dish_id}_mask_occluded.png`` — occluded mask
- ``{dish_id}_obb_all.txt`` — YOLOv8 OBB annotations
- ``{dish_id}_occlusion_meta.json`` — per-occlusion details
- ``occlusion_summary.json`` — pipeline-level summary
"""

import os
import glob
import json
import shutil
import multiprocessing as mp
from datetime import datetime
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from pyrootsim.discontinuity.cut_2cc import (
    CONFIG_TO_CATEGORY,
    place_2cc_occlusions,
    GlobalOcclusionTracker,
)
from pyrootsim.discontinuity.cut_3cc import (
    place_3cc_occlusions,
)
from pyrootsim.discontinuity.cut_4cc import (
    place_4cc_occlusions,
)
from pyrootsim.discontinuity.build_roots import build_roots_fast

# ===================================================================
# CANVAS EDGE PROTECTION
# ===================================================================

CANVAS_EDGE_MARGIN = 18

# ===================================================================
# DENSITY TIERS (0 % skip — relies on pre-split "To_Occlude" folder)
# ===================================================================

DENSITY_TIERS = {
    "skip":    {"prob": 0.00, "2cc": (0, 0), "3cc": (0, 0), "4cc": (0, 0)},
    "minimal": {"prob": 0.13, "2cc": (2, 3), "3cc": (0, 1), "4cc": (0, 0)},
    "light":   {"prob": 0.22, "2cc": (2, 4), "3cc": (1, 1), "4cc": (0, 0)},
    "medium":  {"prob": 0.30, "2cc": (3, 5), "3cc": (1, 2), "4cc": (0, 1)},
    "heavy":   {"prob": 0.22, "2cc": (4, 6), "3cc": (1, 2), "4cc": (1, 1)},
    "extreme": {"prob": 0.13, "2cc": (5, 7), "3cc": (2, 3), "4cc": (1, 1)},
}


# ===================================================================
# HELPERS
# ===================================================================

def _sample_density_tier(rng):
    """Sample a density tier name from the configured probabilities."""
    tiers = list(DENSITY_TIERS.keys())
    probs = np.array([DENSITY_TIERS[t]["prob"] for t in tiers])
    probs /= probs.sum()
    return rng.choice(tiers, p=probs)


def _sample_targets(tier, rng):
    """Sample per-CC-type target counts for a given density tier."""
    cfg = DENSITY_TIERS[tier]
    targets = {}
    for cc_type in ["2cc", "3cc", "4cc"]:
        lo, hi = cfg[cc_type]
        targets[cc_type] = int(rng.integers(lo, hi + 1)) if hi > 0 else 0
    return targets


def _resolve_config_name(meta_path):
    """Extract primary config name from dish metadata JSON."""
    if not os.path.exists(meta_path):
        return "04_Medium_Kinky_Noisy"
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        plants = meta.get("plants", [])
        if plants:
            return plants[0].get("primary_config", "04_Medium_Kinky_Noisy")
    except Exception:
        pass
    return "04_Medium_Kinky_Noisy"


# ===================================================================
# POST-HOC SURGICAL FILTER (2cc)
# ===================================================================

def _filter_and_heal_2cc(original_mask, current_mask, occs_2cc):
    """Heal phantom 2cc cuts that don't produce a true global CC increase.

    For each 2cc occlusion, temporarily heals the cut on the full dish,
    counts global CCs, re-applies the cut, and checks that the delta is
    exactly +1. Phantom cuts are permanently healed.

    Returns
    -------
    tuple of (list, int, numpy.ndarray)
        ``(valid_occlusions, fake_dropped_count, updated_mask)``
    """
    valid_occs = []
    fake_dropped = 0
    h, w = original_mask.shape

    for occ in occs_2cc:
        erase_pixels = occ.get("erase_pixels")
        if not erase_pixels:
            valid_occs.append(occ)
            continue

        # Temporarily heal the cut
        for py, px in erase_pixels:
            if 0 <= py < h and 0 <= px < w:
                current_mask[py, px] = original_mask[py, px]

        # Global "before" count
        bin_before = (current_mask > 0).astype(np.uint8) * 255
        nl_before, _ = cv2.connectedComponents(bin_before, connectivity=8)

        # Re-apply the cut
        for py, px in erase_pixels:
            if 0 <= py < h and 0 <= px < w:
                current_mask[py, px] = 0

        # Global "after" count
        bin_after = (current_mask > 0).astype(np.uint8) * 255
        nl_after, _ = cv2.connectedComponents(bin_after, connectivity=8)

        if (nl_after - nl_before) == 1:
            valid_occs.append(occ)
        else:
            # Permanent heal — phantom cut
            for py, px in erase_pixels:
                if 0 <= py < h and 0 <= px < w:
                    current_mask[py, px] = original_mask[py, px]
            fake_dropped += 1

    return valid_occs, fake_dropped, current_mask


# ===================================================================
# POST-HOC OBB VALIDATION (all CC types)
# ===================================================================

def _validate_all_obbs(occluded_mask, occlusions_meta, original_mask=None):
    """Re-check every OBB after all occlusions have been placed.

    For 3cc / 4cc: CC count inside the OBB must match ``expected_cc``.

    For 2cc (both tests must pass):
      - Test A: OBB must contain >= 2 CCs on the final occluded mask.
      - Test B: global heal/re-cut delta must be exactly +1.

    Parameters
    ----------
    occluded_mask : numpy.ndarray
        Final occluded mask (H, W), uint8 0/255.
    occlusions_meta : list of dict
        Per-occlusion metadata.
    original_mask : numpy.ndarray or None
        Pre-occlusion mask. Required for 2cc heal/re-cut test.

    Returns
    -------
    list of dict
        Validation results, same order as *occlusions_meta*.
    """
    results = []
    h, w = occluded_mask.shape

    for occ in occlusions_meta:
        obb_xy = occ.get("obb_xy")
        expected_cc = occ.get("expected_cc", occ.get("cc_in_obb", -1))
        occ_type = occ.get("occlusion_type", occ.get("type", "unknown"))
        cc_type = occ.get("cc_type", "2cc")

        if obb_xy is None:
            results.append({
                "type": occ_type, "expected": expected_cc,
                "actual": -1, "valid": False, "reason": "no_obb",
            })
            continue

        # Rasterize OBB mask
        obb_box = np.array(obb_xy, dtype=np.float32)
        bi = obb_box.astype(np.int32).reshape(-1, 1, 2)
        bm = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(bm, [bi], 255)
        inside = (bm > 0) & (occluded_mask > 0)
        nl, _ = cv2.connectedComponents(inside.astype(np.uint8), connectivity=8)
        actual_cc = nl - 1

        # 3cc / 4cc: simple CC count match
        if cc_type != "2cc":
            results.append({
                "type": occ_type, "expected": expected_cc,
                "actual": actual_cc, "valid": actual_cc == expected_cc,
            })
            continue

        # 2cc Test A: at least 2 CCs inside OBB
        if actual_cc < 2:
            erase_pixels = occ.get("erase_pixels")
            if erase_pixels is not None and original_mask is not None:
                for py, px in erase_pixels:
                    if 0 <= py < h and 0 <= px < w:
                        occluded_mask[py, px] = original_mask[py, px]
            results.append({
                "type": occ_type, "expected": expected_cc,
                "actual": actual_cc, "valid": False,
                "reason": "fewer_than_2cc_in_obb",
            })
            continue

        # 2cc Test B: global heal/re-cut delta == +1
        erase_pixels = occ.get("erase_pixels")
        if erase_pixels is None or original_mask is None:
            results.append({
                "type": occ_type, "expected": expected_cc,
                "actual": actual_cc, "valid": True,
                "reason": "no_erase_pixels_or_original",
            })
            continue

        # Heal
        for py, px in erase_pixels:
            if 0 <= py < h and 0 <= px < w:
                occluded_mask[py, px] = original_mask[py, px]

        bin_before = (occluded_mask > 0).astype(np.uint8) * 255
        nl_before, _ = cv2.connectedComponents(bin_before, connectivity=8)

        # Re-cut
        for py, px in erase_pixels:
            if 0 <= py < h and 0 <= px < w:
                occluded_mask[py, px] = 0

        bin_after = (occluded_mask > 0).astype(np.uint8) * 255
        nl_after, _ = cv2.connectedComponents(bin_after, connectivity=8)

        delta = nl_after - nl_before
        if delta == 1:
            results.append({
                "type": occ_type, "expected": expected_cc,
                "actual": actual_cc, "valid": True,
            })
        else:
            # Phantom — permanently heal
            for py, px in erase_pixels:
                if 0 <= py < h and 0 <= px < w:
                    occluded_mask[py, px] = original_mask[py, px]
            results.append({
                "type": occ_type, "expected": expected_cc,
                "actual": actual_cc, "valid": False,
                "reason": f"global_delta_{delta}_not_1",
            })

    return results


# ===================================================================
# METADATA CONVERTERS
# ===================================================================

def _occ_to_meta(occ, cc_type, expected_cc):
    """Convert a 2cc or 3cc occlusion dict to serialisable metadata."""
    meta = {
        "cc_type": cc_type,
        "expected_cc": expected_cc,
        "tid": int(occ.get("tid", 0)),
        "plant_id": int(occ.get("plant_id", 0)),
        "root_type": occ.get("root_type", "unknown"),
        "occlusion_type": occ.get("occlusion_type", cc_type),
        "cut_length": occ.get("cut_length", occ.get("primary_cut_length", 0)),
        "size_bin": occ.get("size_bin", "unknown"),
        "cc_in_obb": occ.get("cc_in_obb", -1),
        "is_top_tip": occ.get("is_top_tip", occ.get("is_top_tip_target", False)),
        "is_lateral_base": occ.get("is_lateral_base", False),
        "obb_xy": (
            [[float(p[0]), float(p[1])] for p in occ["obb_box"]]
            if occ.get("obb_box") is not None
            else None
        ),
    }
    # Carry erase_pixels through for the post-hoc 2cc heal/re-cut validator
    if occ.get("erase_pixels") is not None:
        meta["erase_pixels"] = occ["erase_pixels"]
    return meta


def _occ_to_meta_4cc(occ):
    """Convert a 4cc occlusion dict to serialisable metadata."""
    return {
        "cc_type": "4cc",
        "expected_cc": 4,
        "type": occ.get("type", "4cc"),
        "tids": occ.get("tids", []),
        "plant_id_a": occ.get("plant_id_a", 0),
        "plant_id_b": occ.get("plant_id_b", 0),
        "occlusion_type": occ.get("type", "4cc"),
        "cut_length_a": occ.get("cut_length_a", 0),
        "cut_length_b": occ.get("cut_length_b", 0),
        "cc_in_obb": occ.get("cc_in_obb", -1),
        "is_top_tip": occ.get("is_top_tip", False),
        "obb_xy": (
            [[float(p[0]), float(p[1])] for p in occ["obb_box"]]
            if occ.get("obb_box") is not None
            else None
        ),
    }


# ===================================================================
# SAVE DISH OUTPUT
# ===================================================================

def _save_dish_output(output_dir, dish_id, occluded_mask, obb_lines,
                      occlusions_meta, seed, tier, config_name, category,
                      targets, placed):
    """Write occluded mask, OBB annotations, and metadata for one dish."""
    os.makedirs(output_dir, exist_ok=True)
    Image.fromarray(occluded_mask).save(
        os.path.join(output_dir, f"{dish_id}_mask_occluded.png")
    )

    with open(os.path.join(output_dir, f"{dish_id}_obb_all.txt"), "w") as f:
        for line in obb_lines:
            f.write(line + "\n")

    meta = {
        "dish_id": dish_id,
        "config": config_name,
        "category": category,
        "density_tier": tier,
        "targets": targets,
        "placed": placed,
        "total_placed": sum(placed.values()) if placed else 0,
        "occlusions": occlusions_meta,
        "seed": int(seed),
        "generation_timestamp": datetime.now().isoformat(),
    }

    with open(
        os.path.join(output_dir, f"{dish_id}_occlusion_meta.json"), "w"
    ) as f:
        json.dump(meta, f, indent=2)


# ===================================================================
# PROCESS SINGLE DISH
# ===================================================================

def _process_single_dish(task):
    """Process one dish through the full 2cc → 3cc → 4cc pipeline.

    Called by :func:`run_occlusion_pipeline` either directly or via
    a multiprocessing pool.

    Parameters
    ----------
    task : tuple
        ``(mask_path, output_dir, base_seed)``

    Returns
    -------
    dict
        Result dictionary with placement counts, validation stats, etc.
    """
    mask_path, output_dir, base_seed = task

    dish_id = os.path.basename(mask_path).replace("_mask.png", "")
    labels_path = mask_path.replace("_mask.png", "_labels.png")
    overlap_path = mask_path.replace("_mask.png", "_overlap.png")
    meta_path = mask_path.replace("_mask.png", "_metadata.json")

    result = {
        "dish_id": dish_id,
        "success": False,
        "tier": "unknown",
        "targets": {},
        "placed": {},
        "total_placed": 0,
        "validation_ok": 0,
        "validation_bad": 0,
        "fake_2cc_dropped": 0,
        "skipped": False,
    }

    try:
        if not os.path.exists(labels_path):
            result["error"] = "missing_labels"
            return result

        canvas_labels = np.array(Image.open(labels_path))
        overlap_labels = (
            np.array(Image.open(overlap_path))
            if os.path.exists(overlap_path)
            else None
        )

        config_name = _resolve_config_name(meta_path)
        category = CONFIG_TO_CATEGORY.get(config_name, "medium")

        seed = base_seed + abs(hash(dish_id)) % 10 ** 7
        rng = np.random.default_rng(seed)

        tier = _sample_density_tier(rng)
        result["tier"] = tier

        if tier == "skip":
            result["skipped"] = True
            result["targets"] = {"2cc": 0, "3cc": 0, "4cc": 0}
            result["placed"] = {"2cc": 0, "3cc": 0, "4cc": 0}
            canvas_mask = (canvas_labels > 0).astype(np.uint8) * 255
            _save_dish_output(
                output_dir, dish_id, canvas_mask, [], [], seed, tier,
                config_name, category, {}, {},
            )
            result["success"] = True
            return result

        targets = _sample_targets(tier, rng)
        result["targets"] = targets

        roots = build_roots_fast(canvas_labels, overlap_labels)

        h, w = canvas_labels.shape
        tracker = GlobalOcclusionTracker(
            (h, w), canvas_edge_margin=CANVAS_EDGE_MARGIN
        )

        # Save exact original state for healing
        original_mask = (canvas_labels > 0).astype(np.uint8) * 255
        current_mask = original_mask.copy()

        # Phase 1: 2cc
        occs_2cc, current_mask = place_2cc_occlusions(
            roots, canvas_labels, overlap_labels, category, rng, targets["2cc"],
            tracker=tracker, current_mask=current_mask,
        )

        # Phase 2: 3cc
        occs_3cc, current_mask, _ = place_3cc_occlusions(
            roots, canvas_labels, overlap_labels, category, rng, targets["3cc"],
            tracker=tracker, current_mask=current_mask,
        )

        # Phase 3: 4cc
        occs_4cc, current_mask = place_4cc_occlusions(
            roots, canvas_labels, overlap_labels, category, rng, targets["4cc"],
            tracker=tracker, current_mask=current_mask,
        )

        # Post-hoc surgical filter for 2cc
        occs_2cc, fake_2cc_dropped, current_mask = _filter_and_heal_2cc(
            original_mask, current_mask, occs_2cc
        )
        result["fake_2cc_dropped"] = fake_2cc_dropped

        # Build clean metadata
        all_occlusions_meta = []
        for occ in occs_2cc:
            all_occlusions_meta.append(_occ_to_meta(occ, "2cc", 2))
        for occ in occs_3cc:
            all_occlusions_meta.append(_occ_to_meta(occ, "3cc", 3))
        for occ in occs_4cc:
            all_occlusions_meta.append(_occ_to_meta_4cc(occ))

        # Post-hoc OBB validation (all CC types)
        val_results = _validate_all_obbs(
            current_mask, all_occlusions_meta, original_mask=original_mask
        )

        clean_meta = []
        clean_obb_lines = []
        dropped = 0

        for occ_meta, val in zip(all_occlusions_meta, val_results):
            occ_meta["post_validation"] = val
            if val["valid"]:
                clean_meta.append(occ_meta)
                if occ_meta.get("obb_xy") is not None:
                    cc_type = occ_meta.get("cc_type", "2cc")
                    class_id = {"2cc": "0", "3cc": "1", "4cc": "2"}.get(
                        cc_type, "0"
                    )
                    obb_xy = occ_meta["obb_xy"]
                    h_img, w_img = current_mask.shape
                    coords = " ".join(
                        f"{obb_xy[i][0] / w_img:.6f} {obb_xy[i][1] / h_img:.6f}"
                        for i in range(4)
                    )
                    clean_obb_lines.append(f"{class_id} {coords}")
            else:
                dropped += 1

        placed_clean = {"2cc": 0, "3cc": 0, "4cc": 0}
        for om in clean_meta:
            cc = om.get("cc_type", "2cc")
            if cc in placed_clean:
                placed_clean[cc] += 1

        result["validation_ok"] = len(clean_meta)
        result["validation_bad"] = dropped
        result["validation_dropped"] = dropped
        result["placed"] = placed_clean
        result["total_placed"] = sum(placed_clean.values())

        # Strip erase_pixels from metadata before saving
        save_meta = []
        for om in clean_meta:
            om_save = {k: v for k, v in om.items() if k != "erase_pixels"}
            save_meta.append(om_save)

        _save_dish_output(
            output_dir, dish_id, current_mask,
            clean_obb_lines, save_meta,
            seed, tier, config_name, category, targets, placed_clean,
        )

        # Copy original files alongside
        shutil.copy2(
            mask_path, os.path.join(output_dir, os.path.basename(mask_path))
        )
        if os.path.exists(labels_path):
            shutil.copy2(
                labels_path,
                os.path.join(output_dir, os.path.basename(labels_path)),
            )
        if overlap_path and os.path.exists(overlap_path):
            shutil.copy2(
                overlap_path,
                os.path.join(output_dir, os.path.basename(overlap_path)),
            )
        if os.path.exists(meta_path):
            shutil.copy2(
                meta_path,
                os.path.join(output_dir, os.path.basename(meta_path)),
            )

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


# ===================================================================
# SUMMARY & REPORTING
# ===================================================================

def _build_summary(results, input_dir, output_dir, base_seed, num_workers,
                   start_time, end_time, elapsed):
    """Aggregate per-dish results into a pipeline summary dict."""
    ok = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    tier_dist = defaultdict(int)
    for r in results:
        tier_dist[r["tier"]] += 1

    total_occ_counts = [r["total_placed"] for r in ok]
    occ_histogram = defaultdict(int)
    for c in total_occ_counts:
        occ_histogram[c] += 1

    cc_placed = {"2cc": 0, "3cc": 0, "4cc": 0}
    cc_targeted = {"2cc": 0, "3cc": 0, "4cc": 0}
    for r in ok:
        for cc in ["2cc", "3cc", "4cc"]:
            cc_placed[cc] += r["placed"].get(cc, 0)
            cc_targeted[cc] += r["targets"].get(cc, 0)

    total_val_ok = sum(r.get("validation_ok", 0) for r in ok)
    total_val_bad = sum(r.get("validation_bad", 0) for r in ok)
    total_dropped = sum(r.get("validation_dropped", 0) for r in ok)
    total_fake_2cc = sum(r.get("fake_2cc_dropped", 0) for r in ok)

    skipped = sum(1 for r in ok if r.get("skipped", False))

    return {
        "pipeline_version": "pyrootsim",
        "input_dir": input_dir,
        "output_dir": output_dir,
        "base_seed": base_seed,
        "num_workers": num_workers,
        "canvas_edge_margin": CANVAS_EDGE_MARGIN,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "dishes_per_second": round(len(ok) / max(elapsed, 0.1), 2),
        "total_dishes": len(results),
        "successful": len(ok),
        "failed": len(failed),
        "skipped_no_occlusion": skipped,
        "tier_distribution": dict(tier_dist),
        "cc_targeted": cc_targeted,
        "cc_placed": cc_placed,
        "cc_fill_rate": {
            cc: round(cc_placed[cc] / max(cc_targeted[cc], 1), 3)
            for cc in ["2cc", "3cc", "4cc"]
        },
        "total_occlusions": sum(cc_placed.values()),
        "avg_occlusions_per_dish": round(
            sum(cc_placed.values()) / max(len(ok) - skipped, 1), 2
        ),
        "validation_passed": total_val_ok,
        "validation_dropped": total_dropped,
        "fake_2cc_dropped": total_fake_2cc,
        "validation_pass_rate": round(
            total_val_ok / max(total_val_ok + total_dropped, 1), 4
        ),
        "occlusion_count_histogram": dict(sorted(occ_histogram.items())),
        "errors": [
            {"dish_id": r["dish_id"], "error": r.get("error", "unknown")}
            for r in failed[:20]
        ],
    }


def _print_report(summary):
    """Print a human-readable pipeline report to stdout."""
    print(f"\n{'=' * 65}")
    print("pyrootsim — OCCLUSION PIPELINE COMPLETE")
    print(f"{'=' * 65}")
    print(
        f"Time:           {summary['elapsed_seconds']:.1f}s "
        f"({summary['dishes_per_second']:.1f} dishes/sec)"
    )
    print(f"Dishes:         {summary['successful']} / {summary['total_dishes']}")
    print(f"Failed:         {summary['failed']}")
    print(f"Skipped (0 occ):{summary['skipped_no_occlusion']}")
    print()

    print(f"{'CC Type':<8} {'Targeted':>10} {'Placed':>10} {'Fill Rate':>10}")
    print(f"{'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10}")
    for cc in ["2cc", "3cc", "4cc"]:
        print(
            f"{cc:<8} {summary['cc_targeted'][cc]:>10} "
            f"{summary['cc_placed'][cc]:>10} "
            f"{summary['cc_fill_rate'][cc]:>10.1%}"
        )
    print(
        f"{'TOTAL':<8} {sum(summary['cc_targeted'].values()):>10} "
        f"{summary['total_occlusions']:>10}"
    )
    print(f"\nAvg occlusions/dish: {summary['avg_occlusions_per_dish']:.1f}")

    print(f"\nOBB Validation (post-hoc filter):")
    print(f"  Kept:             {summary['validation_passed']}")
    print(f"  Dropped OBB:      {summary['validation_dropped']}")
    print(f"  Dropped Fake 2cc: {summary.get('fake_2cc_dropped', 0)}")
    print(f"  Pass rate:        {summary['validation_pass_rate']:.2%}")

    print(f"\nDensity tier distribution:")
    for tier in ["skip", "minimal", "light", "medium", "heavy", "extreme"]:
        count = summary["tier_distribution"].get(tier, 0)
        total = summary["total_dishes"]
        pct = count / max(total, 1) * 100
        bar = "\u2588" * max(1, int(pct / 2))
        print(f"  {tier:>8}: {count:5d} ({pct:5.1f}%) {bar}")

    print(f"\nOcclusions per dish histogram:")
    hist = summary["occlusion_count_histogram"]
    max_count = max(hist.values()) if hist else 1
    for n_occ in sorted(hist.keys(), key=int):
        freq = hist[n_occ]
        bar = "\u2588" * max(1, int(freq * 40 / max_count))
        print(f"  {n_occ:>3} occs: {freq:5d} {bar}")

    print(f"\nSummary: {summary['output_dir']}/occlusion_summary.json")


# ===================================================================
# MAIN PIPELINE
# ===================================================================

def run_occlusion_pipeline(
    petri_dish_dir,
    output_dir="petri_dishes_occluded",
    base_seed=42,
    num_workers=0,
    clear_existing=True,
    verbose=True,
):
    """Run the full occlusion pipeline on a folder of petri dishes.

    Parameters
    ----------
    petri_dish_dir : str
        Input directory containing ``*_mask.png`` files from the dish
        composer (or a pre-split "to-occlude" subset).
    output_dir : str
        Output directory for occluded dishes and annotations.
    base_seed : int
        Base random seed (per-dish seeds are derived from this).
    num_workers : int
        ``0`` = sequential, ``-1`` = auto (half of physical cores if
        >16 cores, else all), ``N`` = use *N* workers.
    clear_existing : bool
        If True, remove *output_dir* before starting.
    verbose : bool
        Print progress and final report.

    Returns
    -------
    dict
        Pipeline summary (also saved to ``occlusion_summary.json``).
    """
    if clear_existing and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    mask_files = sorted(glob.glob(os.path.join(petri_dish_dir, "*_mask.png")))
    mask_files = [f for f in mask_files if "occluded" not in f]

    if not mask_files:
        print(f"No dishes found in {petri_dish_dir}")
        return {}

    if num_workers == -1:
        try:
            physical = len(os.sched_getaffinity(0))
        except AttributeError:
            physical = mp.cpu_count()
        num_workers = max(1, physical // 2) if physical > 16 else physical
    elif num_workers > 0:
        num_workers = min(num_workers, mp.cpu_count())

    mode_str = f"{num_workers} workers" if num_workers > 0 else "sequential"
    start_time = datetime.now()

    if verbose:
        print(f"{'=' * 65}")
        print("pyrootsim — OCCLUSION PIPELINE")
        print(f"{'=' * 65}")
        print(f"Input:       {petri_dish_dir}")
        print(f"Output:      {output_dir}")
        print(f"Dishes:      {len(mask_files)}")
        print(f"Execution:   {mode_str}")
        print(f"Edge margin: {CANVAS_EDGE_MARGIN}px")
        print(f"{'=' * 65}")

    tasks = [(mf, output_dir, base_seed) for mf in mask_files]

    if num_workers > 0:
        with mp.Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_process_single_dish, tasks),
                    total=len(tasks),
                    desc="Occluding dishes",
                    disable=not verbose,
                    smoothing=0.02,
                )
            )
    else:
        results = []
        for task in tqdm(tasks, desc="Occluding dishes", disable=not verbose):
            results.append(_process_single_dish(task))

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    summary = _build_summary(
        results, petri_dish_dir, output_dir, base_seed, num_workers,
        start_time, end_time, elapsed,
    )

    summary_path = os.path.join(output_dir, "occlusion_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        _print_report(summary)

    return summary