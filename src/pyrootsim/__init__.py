"""
PyRootSim — Synthetic Root System Architecture Simulator.

Generates annotated synthetic datasets of *Arabidopsis thaliana* root
system architectures for machine learning research. The pipeline
produces binary masks, per-root label images, skeleton data, and
oriented bounding box (OBB) annotations ready for tasks such as
discontinuity detection, inpainting, instance segmentation, and root
graph classification.

Default configurations approximate *Arabidopsis thaliana* morphology
based on visual references. They have **not** been validated by a plant
scientist and should not be treated as biologically calibrated.

Subpackages
-----------
roots
    Individual root generation (primary + lateral).
dataset
    Batch orchestration and stratified train/val/test splitting.
dish
    Petri dish composition and visualisation.
discontinuity
    Controlled occlusion injection (2cc, 3cc, 4cc cuts) with
    post-hoc validation.

Dedicated to my grandfather, António Ribeiro.
"""

__version__ = "0.1.0"