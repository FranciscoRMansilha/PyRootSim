"""
Root system configuration parameters for PyRootSim.
 
This module is the single source of truth for all configuration constants
used by the primary and lateral root generators. The default values were
hand-tuned to approximate *Arabidopsis thaliana* root morphology based on
visual reference images. They have **not** been validated by a plant
scientist (yet ;) and should not be treated as biologically calibrated
measurements.
 
This module defines:
 
- Frequency layer presets that control root path curvature
- Primary root morphology configurations (12 named presets)
- Lateral root mode definitions per root-length category
- Width, taper, pinch, and skew defaults for both primary and lateral roots
- Top-tip lateral configuration
- Lateral emergence zone and spacing parameters
- Left/right side distribution patterns
 
All dictionaries and constants are importable by name. Nothing in this
module has side effects — it is pure configuration data.

FUTURE STEP: creating a `SpeciesProfile` dataclass that mirrors the structure below, writing a small validation layer 
(range checks, type checks, making sure probabilities sum to 1 where they should), adding `to_yaml`/`from_yaml` serialisation,
writing a simple registry with `load_profile`, `register_profile`, `list_profiles`, 
and then wiring the generators to accept a profile object as an alternative to the current config name string.

"""

# =========================================================================
# Frequency Layer Presets
# =========================================================================
# Each preset is a list of (frequency, amplitude) tuples that are summed
# to build the angular noise signal driving root path curvature.

BASE_FREQS = [
    (0.015, 0.12),
    (0.040, 0.08),
    (0.150, 0.06),
    (0.250, 0.05),
]
"""Multi-scale noise: broad curves + medium wiggles + fine texture."""

SMOOTH_FREQS = [
    (0.010, 0.20),
    (0.030, 0.10),
]
"""Low-frequency only: gentle, sweeping curves."""

LONG_WAVE_FREQS = [
    (0.005, 0.30),
    (0.015, 0.15),
]
"""Very low frequency: large-scale bends for extra-long roots."""


# =========================================================================
# Primary Root — Taper Defaults
# =========================================================================

DEFAULT_PRIMARY_TAPER_CONFIG = {
    "no_taper_prob": 0.38,
    "mild_taper_prob": 0.32,
    "strong_taper_prob": 0.30,
    "mild_taper_start_range": (0.65, 0.80),
    "mild_taper_end_fraction_range": (0.35, 0.55),
    "strong_taper_start_range": (0.50, 0.72),
    "strong_taper_end_fraction_range": (0.05, 0.25),
    "thin_prob": 0.38,
    "thin_scale_range": (0.25, 0.65),
}
"""Controls how primary root width tapers toward the tip."""


# =========================================================================
# Primary Root — Pinch Defaults
# =========================================================================

DEFAULT_PRIMARY_PINCH_CONFIG = {
    "pinch_enabled_prob": 0.55,
    "count_range": (1, 4),
    "depth_factor_range": (0.30, 0.70),
    "duration_frac_range": (0.03, 0.12),
    "sharpness_range": (1.5, 3.0),
    "forbidden_start_pct": 0.05,
    "forbidden_end_pct": 0.08,
}
"""Controls localised width constrictions along the primary root."""


# =========================================================================
# Primary Root — Width Skew Default
# =========================================================================

DEFAULT_PRIMARY_WIDTH_SKEW = (2, 5)
"""(alpha, beta) parameters for the Beta distribution that samples base width.
Lower alpha / higher beta biases toward thinner roots."""


# =========================================================================
# Primary Root Configurations
# =========================================================================
# Each entry defines a complete morphology preset. Keys prefixed with
# numbers are the 12 main configs; sub-keys like 12a/12b are mix sources.

PRIMARY_CONFIGS = {
    "01_Short_Kinky_Noisy": {
        "category": "short",
        "len_mean": 180, "len_std": 40, "freq_layers": BASE_FREQS,
        "kink_prob": 0.05, "kink_amp": 0.35, "artifact_lvl": "standard",
        "width_mean": 4.0, "width_std": 1.0,
        "width_skew": (2, 4),
    },
    "02_Short_Smooth_Clean": {
        "category": "short",
        "len_mean": 180, "len_std": 40, "freq_layers": SMOOTH_FREQS,
        "kink_prob": 0.0, "kink_amp": 0.0, "artifact_lvl": "low",
        "width_mean": 3.5, "width_std": 0.8,
        "width_skew": (2, 4),
    },
    "03_Short_Kinky_Smooth": {
        "category": "short",
        "len_mean": 180, "len_std": 40, "freq_layers": BASE_FREQS,
        "kink_prob": 0.05, "kink_amp": 0.20,
        "artifact_lvl": "low",
        "width_mean": 3.8, "width_std": 0.9,
        "width_skew": (2, 4),
    },
    "04_Medium_Kinky_Noisy": {
        "category": "medium",
        "len_mean": 800, "len_std": 100, "freq_layers": BASE_FREQS,
        "kink_prob": 0.05, "kink_amp": 0.35, "artifact_lvl": "standard",
        "width_mean": 5.0, "width_std": 1.2,
        "width_skew": (2, 5),
    },
    "05_Medium_Smooth_Snake": {
        "category": "medium",
        "len_mean": 800, "len_std": 100, "freq_layers": SMOOTH_FREQS,
        "kink_prob": 0.0, "kink_amp": 0.0, "artifact_lvl": "standard",
        "width_mean": 5.0, "width_std": 1.0,
        "width_skew": (2, 5),
    },
    "06_Medium_Clean_GroundTruth": {
        "category": "medium",
        "len_mean": 800, "len_std": 100, "freq_layers": BASE_FREQS,
        "kink_prob": 0.05, "kink_amp": 0.35, "artifact_lvl": "none",
        "width_mean": 5.0, "width_std": 0.5,
        "width_skew": (2, 4),
        "taper_override": {
            "no_taper_prob": 1.0,
            "mild_taper_prob": 0.0,
            "strong_taper_prob": 0.0,
            "thin_prob": 0.0,
        },
        "pinch_override": {
            "pinch_enabled_prob": 0.0,
        },
    },
    "07_Long_Kinky_Noisy": {
        "category": "long",
        "len_mean": 1400, "len_std": 150, "freq_layers": BASE_FREQS,
        "kink_prob": 0.04, "kink_amp": 0.35, "artifact_lvl": "standard",
        "width_mean": 5.5, "width_std": 1.3,
        "width_skew": (2, 5),
    },
    "08_Long_Sweeping_Curves": {
        "category": "long",
        "len_mean": 1400, "len_std": 150, "freq_layers": LONG_WAVE_FREQS,
        "kink_prob": 0.0, "kink_amp": 0.0, "artifact_lvl": "standard",
        "width_mean": 5.5, "width_std": 1.2,
        "width_skew": (2, 5),
    },
    "09_Long_Smooth_Static": {
        "category": "long",
        "len_mean": 1400, "len_std": 150, "freq_layers": SMOOTH_FREQS,
        "kink_prob": 0.0, "kink_amp": 0.0, "artifact_lvl": "low",
        "width_mean": 5.5, "width_std": 1.0,
        "width_skew": (2, 5),
    },
    "10_ExtraLong_Hybrid": {
        "category": "extra_long",
        "len_mean": 1800, "len_std": 200, "freq_layers": LONG_WAVE_FREQS,
        "kink_prob": 0.02, "kink_amp": 0.45, "artifact_lvl": "standard",
        "width_mean": 6.5, "width_std": 1.8,
        "width_skew": (1.2, 8),
    },
    "11_ExtraLong_Curvy_Clean": {
        "category": "extra_long",
        "len_mean": 1800, "len_std": 200, "freq_layers": LONG_WAVE_FREQS,
        "kink_prob": 0.0, "kink_amp": 0.0, "artifact_lvl": "none",
        "width_mean": 6.5, "width_std": 1.8,
        "width_skew": (1.5, 6),
        "taper_override": {
            "no_taper_prob": 1.0,
            "mild_taper_prob": 0.0,
            "strong_taper_prob": 0.0,
            "thin_prob": 0.0,
        },
        "pinch_override": {
            "pinch_enabled_prob": 0.0,
        },
    },
    "12_ExtraLong_Mixed_Sweeping": {
        "category": "extra_long_12",
        "len_mean": 1800, "len_std": 200,
        "mix_sources": ["12a_ExtraLong_Gentle", "12b_ExtraLong_Sweeping"],
    },
    "12a_ExtraLong_Gentle": {
        "category": "extra_long_12",
        "len_mean": 1800, "len_std": 200, "freq_layers": SMOOTH_FREQS,
        "kink_prob": 0.02, "kink_amp": 0.45, "artifact_lvl": "low",
        "width_mean": 6.5, "width_std": 1.8,
        "width_skew": (1.2, 8),
    },
    "12b_ExtraLong_Sweeping": {
        "category": "extra_long_12",
        "len_mean": 1800, "len_std": 200, "freq_layers": LONG_WAVE_FREQS,
        "kink_prob": 0.0, "kink_amp": 0.0, "artifact_lvl": "none",
        "width_mean": 6.5, "width_std": 1.8,
        "width_skew": (1.5, 6),
        "taper_override": {
            "no_taper_prob": 1.0,
            "mild_taper_prob": 0.0,
            "strong_taper_prob": 0.0,
            "thin_prob": 0.0,
        },
        "pinch_override": {
            "pinch_enabled_prob": 0.0,
        },
    },
}
"""All 12 primary root morphology presets (+ sub-configs for config 12).

Each config specifies: length distribution, frequency layers for path
curvature, kink probability/amplitude, artifact degradation level,
width distribution, and optional taper/pinch overrides.
"""


# =========================================================================
# Lateral Root — Style Mapping
# =========================================================================
# Maps each primary config to a lateral root style (reduced kink/amplitude
# compared to the primary, since laterals are shorter and thinner).

LATERAL_STYLE_MAPPING = {
    "01_Short_Kinky_Noisy":        {"freq_layers": BASE_FREQS,   "kink_prob": 0.025, "kink_amp": 0.175, "artifact_lvl": "standard"},
    "02_Short_Smooth_Clean":       {"freq_layers": SMOOTH_FREQS, "kink_prob": 0.000, "kink_amp": 0.000, "artifact_lvl": "low"},
    "03_Short_Kinky_Smooth":       {"freq_layers": BASE_FREQS,   "kink_prob": 0.025, "kink_amp": 0.100, "artifact_lvl": "low"},
    "04_Medium_Kinky_Noisy":       {"freq_layers": BASE_FREQS,   "kink_prob": 0.025, "kink_amp": 0.175, "artifact_lvl": "standard"},
    "05_Medium_Smooth_Snake":      {"freq_layers": SMOOTH_FREQS, "kink_prob": 0.000, "kink_amp": 0.000, "artifact_lvl": "standard"},
    "06_Medium_Clean_GroundTruth": {"freq_layers": BASE_FREQS,   "kink_prob": 0.000, "kink_amp": 0.000, "artifact_lvl": "none"},
    "07_Long_Kinky_Noisy":         {"freq_layers": BASE_FREQS,   "kink_prob": 0.020, "kink_amp": 0.175, "artifact_lvl": "standard"},
    "08_Long_Sweeping_Curves":     {"freq_layers": SMOOTH_FREQS, "kink_prob": 0.000, "kink_amp": 0.000, "artifact_lvl": "standard"},
    "09_Long_Smooth_Static":       {"freq_layers": SMOOTH_FREQS, "kink_prob": 0.000, "kink_amp": 0.000, "artifact_lvl": "low"},
    "10_ExtraLong_Hybrid":         {"freq_layers": SMOOTH_FREQS, "kink_prob": 0.010, "kink_amp": 0.225, "artifact_lvl": "standard"},
    "11_ExtraLong_Curvy_Clean":    {"freq_layers": SMOOTH_FREQS, "kink_prob": 0.000, "kink_amp": 0.000, "artifact_lvl": "none"},
    "12_ExtraLong_Mixed_Sweeping": {"freq_layers": SMOOTH_FREQS, "kink_prob": 0.010, "kink_amp": 0.225, "artifact_lvl": "low"},
    "12a_ExtraLong_Gentle":        {"freq_layers": SMOOTH_FREQS, "kink_prob": 0.010, "kink_amp": 0.225, "artifact_lvl": "low"},
    "12b_ExtraLong_Sweeping":      {"freq_layers": SMOOTH_FREQS, "kink_prob": 0.000, "kink_amp": 0.000, "artifact_lvl": "none"},
}
"""Lateral root curvature/artifact style, keyed by primary config name."""


# =========================================================================
# Lateral Root — Mode Definitions
# =========================================================================
# Organised by root-length category. Each mode within a category controls
# how many laterals are generated and their geometric properties.

LATERAL_MODES = {
    "short": {
        "A_few_small":      {"count_range": (0, 2), "length_range": (0.05, 0.20), "angle_offset_range": (0.70, 1.10), "drift_range": (0.15, 0.25), "description": "Few small laterals"},
        "B_few_horizontal": {"count_range": (1, 2), "length_range": (0.15, 0.35), "angle_offset_range": (1.20, 1.50), "drift_range": (0.02, 0.08), "description": "Few horizontal laterals"},
        "C_few_arched":     {"count_range": (1, 2), "length_range": (0.10, 0.30), "angle_offset_range": (0.50, 0.80), "drift_range": (0.25, 0.40), "description": "Few arched down laterals"},
        "D_few_mixed":      {"count_range": (0, 3), "length_range": (0.05, 0.35), "angle_offset_range": (0.70, 1.40), "drift_range": (0.05, 0.30), "description": "Few mixed laterals"},
        "E_none":           {"count_range": (0, 0), "length_range": (0.0, 0.0),   "angle_offset_range": (0.70, 1.40), "drift_range": (0.10, 0.20), "description": "No laterals"},
    },
    "medium": {
        "A_few_small":        {"count_range": (2, 4), "length_range": (0.05, 0.20), "angle_offset_range": (0.70, 1.10), "drift_range": (0.15, 0.25), "description": "Few small laterals"},
        "B_few_horizontal":   {"count_range": (2, 4), "length_range": (0.25, 0.45), "angle_offset_range": (1.20, 1.50), "drift_range": (0.08, 0.15), "description": "Few big horizontal"},
        "C_few_arched":       {"count_range": (2, 4), "length_range": (0.20, 0.40), "angle_offset_range": (0.50, 0.80), "drift_range": (0.25, 0.40), "description": "Few arched down"},
        "D_few_mixed":        {"count_range": (2, 4), "length_range": (0.10, 0.40), "angle_offset_range": (0.70, 1.20), "drift_range": (0.10, 0.30), "description": "Few mixed"},
        "E_medium_small":     {"count_range": (5, 6), "length_range": (0.05, 0.20), "angle_offset_range": (0.70, 1.10), "drift_range": (0.15, 0.25), "description": "Medium count small"},
        "F_medium_horizontal":{"count_range": (5, 6), "length_range": (0.25, 0.45), "angle_offset_range": (1.20, 1.50), "drift_range": (0.08, 0.15), "description": "Medium count horizontal"},
        "G_medium_arched":    {"count_range": (5, 6), "length_range": (0.20, 0.40), "angle_offset_range": (0.50, 0.80), "drift_range": (0.25, 0.40), "description": "Medium count arched"},
        "H_medium_mixed":     {"count_range": (5, 6), "length_range": (0.10, 0.45), "angle_offset_range": (0.70, 1.20), "drift_range": (0.10, 0.30), "description": "Medium count mixed"},
        "I_many_small":       {"count_range": (7, 7), "length_range": (0.05, 0.20), "angle_offset_range": (0.70, 1.10), "drift_range": (0.15, 0.25), "description": "Many small"},
        "J_many_horizontal":  {"count_range": (7, 7), "length_range": (0.25, 0.50), "angle_offset_range": (1.20, 1.50), "drift_range": (0.08, 0.15), "description": "Many horizontal"},
        "K_many_arched":      {"count_range": (7, 7), "length_range": (0.20, 0.45), "angle_offset_range": (0.50, 0.80), "drift_range": (0.25, 0.40), "description": "Many arched"},
        "L_many_mixed":       {"count_range": (7, 7), "length_range": (0.10, 0.50), "angle_offset_range": (0.70, 1.20), "drift_range": (0.10, 0.30), "description": "Many mixed"},
    },
    "long": {
        "A_few_small":        {"count_range": (3, 5),   "length_range": (0.05, 0.22), "angle_offset_range": (0.70, 1.10), "drift_range": (0.15, 0.25), "description": "Few small laterals"},
        "B_few_horizontal":   {"count_range": (3, 5),   "length_range": (0.15, 0.40), "angle_offset_range": (1.20, 1.50), "drift_range": (0.08, 0.15), "description": "Few big horizontal"},
        "C_few_arched":       {"count_range": (3, 5),   "length_range": (0.12, 0.38), "angle_offset_range": (0.50, 0.80), "drift_range": (0.25, 0.40), "description": "Few arched down"},
        "D_few_mixed":        {"count_range": (3, 5),   "length_range": (0.08, 0.38), "angle_offset_range": (0.70, 1.20), "drift_range": (0.10, 0.30), "description": "Few mixed"},
        "E_medium_small":     {"count_range": (6, 8),   "length_range": (0.05, 0.22), "angle_offset_range": (0.70, 1.10), "drift_range": (0.15, 0.25), "description": "Medium count small"},
        "F_medium_horizontal":{"count_range": (6, 8),   "length_range": (0.15, 0.40), "angle_offset_range": (1.20, 1.50), "drift_range": (0.08, 0.15), "description": "Medium count horizontal"},
        "G_medium_arched":    {"count_range": (6, 8),   "length_range": (0.12, 0.38), "angle_offset_range": (0.50, 0.80), "drift_range": (0.25, 0.40), "description": "Medium count arched"},
        "H_medium_mixed":     {"count_range": (6, 8),   "length_range": (0.08, 0.40), "angle_offset_range": (0.70, 1.20), "drift_range": (0.10, 0.30), "description": "Medium count mixed"},
        "I_many_small":       {"count_range": (10, 12), "length_range": (0.05, 0.22), "angle_offset_range": (0.70, 1.10), "drift_range": (0.15, 0.25), "description": "Many small"},
        "J_many_horizontal":  {"count_range": (10, 12), "length_range": (0.15, 0.40), "angle_offset_range": (1.20, 1.50), "drift_range": (0.08, 0.15), "description": "Many horizontal"},
        "K_many_arched":      {"count_range": (10, 12), "length_range": (0.12, 0.40), "angle_offset_range": (0.50, 0.80), "drift_range": (0.25, 0.40), "description": "Many arched"},
        "L_many_mixed":       {"count_range": (10, 12), "length_range": (0.08, 0.40), "angle_offset_range": (0.70, 1.20), "drift_range": (0.10, 0.30), "description": "Many mixed"},
    },
    "extra_long": {
        "A_few_small":        {"count_range": (4, 6),   "length_range": (0.05, 0.20), "angle_offset_range": (0.70, 1.10), "drift_range": (0.15, 0.25), "description": "Few small laterals"},
        "B_few_horizontal":   {"count_range": (4, 6),   "length_range": (0.12, 0.35), "angle_offset_range": (1.20, 1.50), "drift_range": (0.08, 0.15), "description": "Few big horizontal"},
        "C_few_arched":       {"count_range": (4, 6),   "length_range": (0.10, 0.32), "angle_offset_range": (0.50, 0.80), "drift_range": (0.25, 0.40), "description": "Few arched down"},
        "D_few_mixed":        {"count_range": (4, 6),   "length_range": (0.05, 0.32), "angle_offset_range": (0.70, 1.20), "drift_range": (0.10, 0.30), "description": "Few mixed"},
        "E_medium_small":     {"count_range": (8, 10),  "length_range": (0.05, 0.20), "angle_offset_range": (0.70, 1.10), "drift_range": (0.15, 0.25), "description": "Medium count small"},
        "F_medium_horizontal":{"count_range": (8, 10),  "length_range": (0.12, 0.35), "angle_offset_range": (1.20, 1.50), "drift_range": (0.08, 0.15), "description": "Medium count horizontal"},
        "G_medium_arched":    {"count_range": (8, 10),  "length_range": (0.10, 0.32), "angle_offset_range": (0.50, 0.80), "drift_range": (0.25, 0.40), "description": "Medium count arched"},
        "H_medium_mixed":     {"count_range": (8, 10),  "length_range": (0.05, 0.35), "angle_offset_range": (0.70, 1.20), "drift_range": (0.10, 0.30), "description": "Medium count mixed"},
        "I_many_small":       {"count_range": (13, 15), "length_range": (0.05, 0.20), "angle_offset_range": (0.70, 1.10), "drift_range": (0.15, 0.25), "description": "Many small"},
        "J_many_horizontal":  {"count_range": (13, 15), "length_range": (0.12, 0.38), "angle_offset_range": (1.20, 1.50), "drift_range": (0.08, 0.15), "description": "Many horizontal"},
        "K_many_arched":      {"count_range": (13, 15), "length_range": (0.10, 0.35), "angle_offset_range": (0.50, 0.80), "drift_range": (0.25, 0.40), "description": "Many arched"},
        "L_many_mixed":       {"count_range": (13, 15), "length_range": (0.05, 0.38), "angle_offset_range": (0.70, 1.20), "drift_range": (0.10, 0.30), "description": "Many mixed"},
    },
    "extra_long_12": {
        "A_few_small":        {"count_range": (4, 6),   "length_range": (0.07, 0.24), "angle_offset_range": (0.70, 1.10), "drift_range": (0.15, 0.25), "description": "Few small laterals"},
        "B_few_horizontal":   {"count_range": (4, 6),   "length_range": (0.15, 0.40), "angle_offset_range": (1.20, 1.50), "drift_range": (0.08, 0.15), "description": "Few big horizontal"},
        "C_few_arched":       {"count_range": (4, 6),   "length_range": (0.13, 0.37), "angle_offset_range": (0.50, 0.80), "drift_range": (0.25, 0.40), "description": "Few arched down"},
        "D_few_mixed":        {"count_range": (4, 6),   "length_range": (0.07, 0.37), "angle_offset_range": (0.70, 1.20), "drift_range": (0.10, 0.30), "description": "Few mixed"},
        "E_medium_small":     {"count_range": (8, 10),  "length_range": (0.07, 0.24), "angle_offset_range": (0.70, 1.10), "drift_range": (0.15, 0.25), "description": "Medium count small"},
        "F_medium_horizontal":{"count_range": (8, 10),  "length_range": (0.15, 0.40), "angle_offset_range": (1.20, 1.50), "drift_range": (0.08, 0.15), "description": "Medium count horizontal"},
        "G_medium_arched":    {"count_range": (8, 10),  "length_range": (0.13, 0.37), "angle_offset_range": (0.50, 0.80), "drift_range": (0.25, 0.40), "description": "Medium count arched"},
        "H_medium_mixed":     {"count_range": (8, 10),  "length_range": (0.07, 0.40), "angle_offset_range": (0.70, 1.20), "drift_range": (0.10, 0.30), "description": "Medium count mixed"},
        "I_many_small":       {"count_range": (13, 15), "length_range": (0.07, 0.24), "angle_offset_range": (0.70, 1.10), "drift_range": (0.15, 0.25), "description": "Many small"},
        "J_many_horizontal":  {"count_range": (13, 15), "length_range": (0.15, 0.43), "angle_offset_range": (1.20, 1.50), "drift_range": (0.08, 0.15), "description": "Many horizontal"},
        "K_many_arched":      {"count_range": (13, 15), "length_range": (0.13, 0.40), "angle_offset_range": (0.50, 0.80), "drift_range": (0.25, 0.40), "description": "Many arched"},
        "L_many_mixed":       {"count_range": (13, 15), "length_range": (0.07, 0.43), "angle_offset_range": (0.70, 1.20), "drift_range": (0.10, 0.30), "description": "Many mixed"},
    },
}
"""Lateral root modes organised by primary root length category.

Each mode controls: count range, length as fraction of primary, emergence
angle offset from the primary skeleton, and gravitropic drift magnitude.
"""


# =========================================================================
# Lateral Root — Emergence Zone
# =========================================================================

LATERAL_EMERGENCE_ZONE = {
    "forbidden_tip_pct": 0.25,
    "zones": [
        {"range": (0.00, 0.15), "weight": 0.10},
        {"range": (0.15, 0.55), "weight": 0.65},
        {"range": (0.55, 0.75), "weight": 0.25},
    ],
}
"""Controls where along the primary skeleton laterals can emerge.
The bottom 25% (tip region) is forbidden; the middle zone is most likely."""


# =========================================================================
# Lateral Root — Left/Right Side Patterns
# =========================================================================

LATERAL_LR_PATTERNS = [
    {"name": "balanced",     "prob": 0.45, "left_frac": (0.45, 0.55), "alternating": False},
    {"name": "alternating",  "prob": 0.25, "left_frac": (0.45, 0.55), "alternating": True},
    {"name": "slight_left",  "prob": 0.10, "left_frac": (0.60, 0.70), "alternating": False},
    {"name": "slight_right", "prob": 0.10, "left_frac": (0.30, 0.40), "alternating": False},
    {"name": "strong_left",  "prob": 0.04, "left_frac": (0.75, 0.85), "alternating": False},
    {"name": "strong_right", "prob": 0.04, "left_frac": (0.15, 0.25), "alternating": False},
    {"name": "all_one_side", "prob": 0.02, "left_frac": (1.0, 1.0),   "alternating": False},
]
"""Distribution patterns for assigning laterals to left or right side."""


# =========================================================================
# Lateral Root — Width Configuration
# =========================================================================

LATERAL_WIDTH_CONFIG = {
    "start_fraction_range": (0.55, 0.95),
    "end_fraction_of_start": 0.75,
    "min_width_pixels_default": 2,
    "min_width_pixels_tapered": 1,
}
"""Controls lateral width relative to the parent primary root width."""


# =========================================================================
# Lateral Root — Spacing Configuration
# =========================================================================

LATERAL_SPACING_CONFIG = {
    "min_same_side_pct": 0.04,
    "min_opposite_side_pct": 0.015,
    "overlap_prob": 0.12,
}
"""Minimum spacing between lateral attachment points along the skeleton."""


# =========================================================================
# Lateral Root — Taper Defaults
# =========================================================================

DEFAULT_LATERAL_TAPER_CONFIG = {
    "no_taper_prob": 0.25,
    "mild_taper_prob": 0.37,
    "strong_taper_prob": 0.38,
    "mild_taper_start_range": (0.60, 0.75),
    "mild_taper_end_fraction_range": (0.30, 0.55),
    "strong_taper_start_range": (0.45, 0.65),
    "strong_taper_end_fraction_range": (0.05, 0.30),
    "thin_prob_normal_parent": 0.30,
    "thin_prob_thin_parent": 0.15,
    "thin_scale_range": (0.35, 0.75),
    "attachment_zone_pixels": 8,
    "attachment_zone_min_fraction": 0.45,
}
"""Controls how lateral root width tapers toward its tip."""

LATERAL_TAPER_DISABLED_CONFIGS = {
    "06_Medium_Clean_GroundTruth",
    "11_ExtraLong_Curvy_Clean",
    "12b_ExtraLong_Sweeping",
}
"""Primary configs where lateral taper is disabled (ground-truth / clean)."""


# =========================================================================
# Lateral Root — Pinch Defaults
# =========================================================================

DEFAULT_LATERAL_PINCH_CONFIG = {
    "pinch_enabled_prob": 0.45,
    "count_range": (1, 3),
    "depth_factor_range": (0.30, 0.70),
    "duration_frac_range": (0.04, 0.15),
    "sharpness_range": (1.5, 3.0),
    "forbidden_start_pct": 0.08,
    "forbidden_end_pct": 0.10,
}
"""Controls localised width constrictions along lateral roots."""

LATERAL_PINCH_DISABLED_CONFIGS = {
    "06_Medium_Clean_GroundTruth",
    "11_ExtraLong_Curvy_Clean",
    "12b_ExtraLong_Sweeping",
}
"""Primary configs where lateral pinch is disabled."""


# =========================================================================
# Lateral Root — Width Skew Defaults
# =========================================================================

DEFAULT_LATERAL_WIDTH_SKEW = (2, 4)
"""(alpha, beta) for the Beta distribution sampling lateral base width."""

LATERAL_WIDTH_SKEW_DISABLED_CONFIGS = {
    "06_Medium_Clean_GroundTruth",
    "11_ExtraLong_Curvy_Clean",
    "12b_ExtraLong_Sweeping",
}
"""Primary configs where lateral width skew is disabled (uniform sampling)."""


# =========================================================================
# Lateral Root — Top-Tip Configuration
# =========================================================================

TOP_TIP_CONFIG = {
    "injection_prob": 0.60,
    "count_weights": {1: 0.65, 2: 0.35},
    "attachment_zone_tip_pct": (0.00, 0.00),
    "attachment_zone_near_pct": (0.02, 0.06),
    "near_tip_extra_prob": 0.30,
    "near_tip_extra_count": {1: 0.80, 2: 0.20},
    "angle_offset_range": (0.80, 1.30),
    "upward_kick_prob": 0.40,
    "upward_kick_length_frac": (0.08, 0.20),
    "upward_kick_angle_boost": (0.15, 0.40),
    "drift_range": (0.15, 0.45),
    "length_range": (0.08, 0.30),
    "width_start_fraction_range": (0.70, 1.0),
    "opposite_side_prob": 0.75,
}
"""Configuration for laterals that emerge at or near the primary root tip."""

TOP_TIP_DISABLED_CONFIGS = {
    "06_Medium_Clean_GroundTruth",
    "11_ExtraLong_Curvy_Clean",
    "12b_ExtraLong_Sweeping",
}
"""Primary configs where top-tip laterals are disabled."""


# =========================================================================
# Lateral Root — Composite Retry Limit
# =========================================================================

MAX_COMPOSITE_RETRIES = 5
"""Maximum re-generation attempts to achieve 8-connectivity in the
combined primary + laterals mask before falling back to largest component."""