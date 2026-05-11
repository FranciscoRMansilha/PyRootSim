# PyRootSim

**Synthetic Root System Architecture Simulator**

PyRootSim is an open-source Python package that generates annotated synthetic datasets of *Arabidopsis thaliana* root system architectures — binary masks, per-root label images, and skeleton data — with configurable discontinuity injection (controlled occlusions producing 2, 3, or 4 connected components with OBB annotations). Ready for machine learning tasks such as root discontinuity detection, discontinuity inpainting, instance segmentation, and root graph classification.

Default configurations approximate *Arabidopsis thaliana* morphology based on visual references. They have **not** been validated by a plant scientist and should not be treated as biologically calibrated.

## Installation

```bash
pip install pyrootsim
```

To run the tutorial notebook:

```bash
pip install pyrootsim[tutorial]
```

## Quick Start

```python
from pyrootsim.roots.primary import generate_single_primary_root
from pyrootsim.roots.lateral import generate_lateral_root_inline

# Generate a single primary root
primary = generate_single_primary_root("05_Medium_Smooth_Snake", root_id=0, seed=42)

# Attach lateral roots
composite = generate_lateral_root_inline(
    primary_data=primary,
    config_name="05_Medium_Smooth_Snake",
    lateral_mode_name="E_medium_small",
    seed=42,
)

print(f"Composite mask shape: {composite['combined_mask'].shape}")
print(f"Lateral count: {composite['metadata']['num_laterals']}")
```

## Pipeline Overview

PyRootSim's full pipeline has six stages:

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `pyrootsim.roots` | Generate individual primary + lateral roots with stochastic paths, variable-width profiles, and configurable artefacts |
| 2 | `pyrootsim.dataset.orchestrator` | Batch-generate across all 12 config × 12 lateral mode combinations |
| 3 | `pyrootsim.dish.composer` | Compose 5-seedling petri dish images simulating NPEC's HADES phenotyping system |
| 4 | `pyrootsim.dataset.splitter` | Stratified split into clean vs. to-occlude sets |
| 5 | `pyrootsim.discontinuity.orchestrator` | Apply 2cc / 3cc / 4cc discontinuities with post-hoc OBB validation |
| 6 | `pyrootsim.dataset.splitter` | Final stratified train / val / test split (70 / 20 / 10) |

## Full Pipeline Example

```python
from pyrootsim.dataset.orchestrator import generate_rsa_dataset
from pyrootsim.dish.composer import generate_all_dishes
from pyrootsim.dataset.splitter import split_clean_vs_occluded, split_train_val_test
from pyrootsim.discontinuity.orchestrator import run_occlusion_pipeline

# 1. Generate individual roots (40 per config × mode cell)
generate_rsa_dataset(output_dir="RSA_dataset", num_per_cell=40, base_seed=42)

# 2. Compose petri dishes
generate_all_dishes(rsa_input_dir="RSA_dataset", output_dir="petri_dishes",
                    num_dishes=800, base_seed=42)

# 3. Split clean vs. occluded
split_clean_vs_occluded(input_dir="petri_dishes", occluded_dir="to_occlude",
                        clean_dir="clean", num_clean=150, seed=42)

# 4. Apply discontinuities
run_occlusion_pipeline(petri_dish_dir="to_occlude",
                       output_dir="occluded", base_seed=42)

# 5. Final train/val/test split
split_train_val_test(clean_csv_path="clean/clean_split_metadata.csv",
                     occluded_csv_path="to_occlude/occluded_split_metadata.csv",
                     clean_img_dir="clean", occluded_img_dir="occluded",
                     output_base_dir="final_dataset", seed=42)
```

See `examples/tutorial.ipynb` for a complete walkthrough with visualisations.

## Root Configurations

PyRootSim ships with 12 primary root configurations across 4 length categories:

| Category | Configs |
|----------|---------|
| Short | `01_Short_Kinky_Noisy`, `02_Short_Smooth_Clean`, `03_Short_Kinky_Smooth` |
| Medium | `04_Medium_Kinky_Noisy`, `05_Medium_Smooth_Snake`, `06_Medium_Clean_GroundTruth` |
| Long | `07_Long_Kinky_Noisy`, `08_Long_Sweeping_Curves`, `09_Long_Smooth_Static` |
| Extra Long | `10_ExtraLong_Hybrid`, `11_ExtraLong_Curvy_Clean`, `12_ExtraLong_Mixed_Sweeping` |

Each category has up to 12 lateral branching modes (few/medium/many × small/horizontal/arched/mixed), plus a "no laterals" mode for short roots.

## Outputs

Each processed dish produces:

- `*_mask.png` — Binary root mask
- `*_labels.png` — Per-root label image (plant_id × 100 + lateral_id)
- `*_overlap.png` — Pixels shared by multiple roots
- `*_metadata.json` — Generation parameters
- `*_mask_occluded.png` — Mask after discontinuity injection
- `*_obb_all.txt` — OBB annotations in YOLOv8 format (class 0 = 2cc, 1 = 3cc, 2 = 4cc)
- `*_occlusion_meta.json` — Per-occlusion details

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use PyRootSim in your research, please cite:

```
@software{pyrootsim,
  author = {Mansilha, Francisco Ribeiro},
  title = {PyRootSim: Synthetic Root System Architecture Simulator},
  year = {2026},
  url = {https://github.com/FranciscoRMansilha/PyRootSim},
}
```

---

*Dedicated to my grandfather, António Ribeiro.*