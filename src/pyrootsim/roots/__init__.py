"""
pyrootsim.roots — Synthetic root system generation.

This subpackage contains modules for generating synthetic root system
architecture masks. The default configurations shipped with PyRootSim
were designed to approximate *Arabidopsis thaliana* root morphology
based on visual reference images and are intended for machine-learning
research. They have **not** been validated by a plant scientist YET ;)
Be careful about it

Submodules:

- ``configs``:   All configuration constants (single source of truth)
- ``primary``:   Primary (tap) root generation
- ``lateral``:   Lateral (branch) root generation and compositing
"""