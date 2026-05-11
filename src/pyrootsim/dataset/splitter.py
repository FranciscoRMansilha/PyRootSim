"""
Dataset splitting utilities for PyRootSim.

Provides two stratified splitting operations used at different stages
of the dataset construction pipeline:

1. :func:`split_clean_vs_occluded` — Splits a generated petri-dish
   dataset into an *occluded* set (dishes that will receive
   discontinuity injection) and a *clean* set, stratified by dominant
   primary configuration.

2. :func:`split_train_val_test` — Takes the occluded and clean sets and
   performs the final 70 / 20 / 10 train / val / test split, stratified
   by both primary configuration **and** occlusion status.  Files are
   copied in parallel for speed.

Both functions produce detailed stratification reports on stdout and
save reference CSVs alongside the copied files.
"""

import os
import shutil
import concurrent.futures

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# =========================================================================
# File Extension Definitions
# =========================================================================

_CLEAN_EXTENSIONS = [
    "_mask.png", "_labels.png", "_overlap.png", "_metadata.json",
]

_OCCLUDED_EXTENSIONS = _CLEAN_EXTENSIONS + [
    "_mask_occluded.png", "_obb_all.txt", "_occlusion_meta.json",
]


# =========================================================================
# Stage 1 — Clean vs. Occluded Split
# =========================================================================

def split_clean_vs_occluded(
    input_dir,
    occluded_dir,
    clean_dir,
    num_clean=1200,
    seed=42,
):
    """Split generated dishes into *occluded* and *clean* sets.

    Uses stratified sampling on ``dominant_primary_config`` so that the
    12 root configurations are evenly distributed across both output
    folders.

    The input directory must contain a ``dish_metadata.csv`` produced by
    the petri-dish generator.

    Args:
        input_dir:    directory containing generated dishes + CSV.
        occluded_dir: destination for dishes to be occluded.
        clean_dir:    destination for clean (unmodified) dishes.
        num_clean:    number of dishes to assign to the clean set.
        seed:         random seed for reproducibility.
    """
    csv_path = os.path.join(input_dir, "dish_metadata.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find {csv_path}. "
            "Make sure the petri-dish generator finished running."
        )

    print("Loading metadata and calculating split...")
    df = pd.read_csv(csv_path)

    df_occ, df_clean = train_test_split(
        df,
        test_size=num_clean,
        stratify=df["dominant_primary_config"],
        random_state=seed,
    )

    # --- Copy files ---
    extensions = ["_mask.png", "_labels.png", "_overlap.png", "_metadata.json"]

    os.makedirs(occluded_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)

    def _copy_dish(dish_id, source_dir, dest_dir):
        for ext in extensions:
            src = os.path.join(source_dir, f"{dish_id}{ext}")
            dst = os.path.join(dest_dir, f"{dish_id}{ext}")
            if os.path.exists(src):
                shutil.copy2(src, dst)

    print(f"\nCopying {len(df_clean)} dishes to clean folder: {clean_dir}")
    for dish_id in tqdm(df_clean["dish_id"], desc="Copying clean"):
        _copy_dish(dish_id, input_dir, clean_dir)

    print(f"\nCopying {len(df_occ)} dishes to occluded folder: {occluded_dir}")
    for dish_id in tqdm(df_occ["dish_id"], desc="Copying occluded"):
        _copy_dish(dish_id, input_dir, occluded_dir)

    # --- Save reference CSVs ---
    df_occ.to_csv(
        os.path.join(occluded_dir, "occluded_split_metadata.csv"), index=False,
    )
    df_clean.to_csv(
        os.path.join(clean_dir, "clean_split_metadata.csv"), index=False,
    )

    # --- Report ---
    _print_clean_occluded_report(df, df_occ, df_clean)


def _print_clean_occluded_report(df_all, df_occ, df_clean):
    """Print a stratification report for the clean/occluded split."""
    orig_counts = df_all["dominant_primary_config"].value_counts().sort_index()
    occ_counts = df_occ["dominant_primary_config"].value_counts().sort_index()
    clean_counts = df_clean["dominant_primary_config"].value_counts().sort_index()

    print("\n" + "=" * 70)
    print("STRATIFIED SPLIT REPORT")
    print("=" * 70)
    print(
        f"{'Configuration':<35} | {'Original':<8}"
        f" | {'To Occlude':<10} | {'Clean':<6}"
    )
    print("-" * 70)

    for config in orig_counts.index:
        o_val = orig_counts.get(config, 0)
        occ_val = occ_counts.get(config, 0)
        c_val = clean_counts.get(config, 0)
        print(f"{config:<35} | {o_val:<8} | {occ_val:<10} | {c_val:<6}")

    print("-" * 70)
    print(
        f"{'TOTAL':<35} | {len(df_all):<8}"
        f" | {len(df_occ):<10} | {len(df_clean):<6}"
    )
    print("=" * 70)
    print("Saved reference CSVs to both output directories.")


# =========================================================================
# Stage 2 — Train / Val / Test Split
# =========================================================================

def _copy_worker(task):
    """Worker: copy all files for a single dish (used by thread pool)."""
    dish_id, status, src_dir, dst_dir = task
    exts = _OCCLUDED_EXTENSIONS if status == "Occluded" else _CLEAN_EXTENSIONS
    for ext in exts:
        src_path = os.path.join(src_dir, f"{dish_id}{ext}")
        dst_path = os.path.join(dst_dir, f"{dish_id}{ext}")
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
    return True


def split_train_val_test(
    clean_csv_path,
    occluded_csv_path,
    clean_img_dir,
    occluded_img_dir,
    output_base_dir,
    seed=42,
    max_workers=64,
):
    """Perform a strict 70 / 20 / 10 stratified train / val / test split.

    Stratification is on both occlusion status **and** dominant primary
    configuration to ensure balanced representation in every split.

    File copying is parallelised with a thread pool.

    Args:
        clean_csv_path:    path to the clean-split reference CSV.
        occluded_csv_path: path to the occluded-split reference CSV.
        clean_img_dir:     directory containing clean dish files.
        occluded_img_dir:  directory containing occluded dish files.
        output_base_dir:   root of the final dataset tree.
        seed:              random seed.
        max_workers:       thread-pool size for parallel copying.
    """
    print("Loading metadata...")
    df_clean = pd.read_csv(clean_csv_path)
    df_clean["is_occluded"] = "Clean"

    df_occ = pd.read_csv(occluded_csv_path)
    df_occ["is_occluded"] = "Occluded"

    df_all = pd.concat([df_occ, df_clean], ignore_index=True)
    df_all["stratify_key"] = (
        df_all["dominant_primary_config"] + "_" + df_all["is_occluded"]
    )

    print(f"Total dishes found: {len(df_all)}")

    # --- Split ---
    df_train, df_temp = train_test_split(
        df_all, test_size=0.30,
        stratify=df_all["stratify_key"], random_state=seed,
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=1 / 3,
        stratify=df_temp["stratify_key"], random_state=seed,
    )

    splits = {"train": df_train, "val": df_val, "test": df_test}

    # --- Parallel file copying ---
    print("\nCopying files into dataset structure...")
    for split_name, df_split in splits.items():
        print(
            f"\nProcessing {split_name.upper()} set"
            f" ({len(df_split)} dishes)..."
        )

        tasks = []
        for row in df_split.itertuples(index=False):
            dish_id = row.dish_id
            status = row.is_occluded
            src_dir = occluded_img_dir if status == "Occluded" else clean_img_dir
            dst_dir = os.path.join(output_base_dir, split_name, status.lower())
            os.makedirs(dst_dir, exist_ok=True)
            tasks.append((dish_id, status, src_dir, dst_dir))

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
        ) as executor:
            list(tqdm(
                executor.map(_copy_worker, tasks),
                total=len(tasks),
                desc=f"Copying {split_name}",
            ))

        # Save per-split CSV
        split_csv_path = os.path.join(
            output_base_dir, split_name, f"{split_name}_metadata.csv",
        )
        df_split.drop(columns=["stratify_key"]).to_csv(
            split_csv_path, index=False,
        )

    # --- Report ---
    _print_train_val_test_report(df_all, splits)


def _print_train_val_test_report(df_all, splits):
    """Print a detailed stratification report for the final split."""
    print("\n" + "=" * 85)
    print(f"{'FINAL DATASET STRATIFICATION REPORT':^85}")
    print("=" * 85)

    print("\n1. HIGH-LEVEL SPLIT (Target: 70% Train / 20% Val / 10% Test)")
    print("-" * 50)
    for name, df_s in splits.items():
        pct = len(df_s) / len(df_all) * 100
        n_occ = len(df_s[df_s["is_occluded"] == "Occluded"])
        n_cln = len(df_s[df_s["is_occluded"] == "Clean"])
        print(
            f"{name.upper():<7} | {len(df_s):>5} dishes ({pct:>4.1f}%)"
            f" | Occluded: {n_occ:>4} | Clean: {n_cln:>4}"
        )

    print("\n2. CONFIGURATION DISTRIBUTION ACROSS SPLITS")
    print("-" * 85)
    configs = sorted(df_all["dominant_primary_config"].unique())
    print(
        f"{'Config Name':<30} | {'Train (O/C)':<11}"
        f" | {'Val (O/C)':<11} | {'Test (O/C)':<11} | {'Total'}"
    )
    print("-" * 85)

    for c in configs:
        row_str = f"{c[:28]:<30} | "
        total_c = 0
        for split_name in ("train", "val", "test"):
            df_s = splits[split_name]
            sub = df_s[df_s["dominant_primary_config"] == c]
            o_count = len(sub[sub["is_occluded"] == "Occluded"])
            c_count = len(sub[sub["is_occluded"] == "Clean"])
            row_str += f"{o_count:>4}/{c_count:<4} | "
            total_c += o_count + c_count
        row_str += f"{total_c:>5}"
        print(row_str)

    print("=" * 85)