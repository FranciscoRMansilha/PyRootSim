"""
pyrootsim.dataset — Dataset generation, orchestration, and splitting.

This subpackage contains modules for building complete root-system
architecture datasets:

- ``orchestrator``: Generates a full config × lateral-mode grid of roots
  with optional multiprocessing support.
- ``splitter``:     Stratified train/val/test splitting for generated
  petri-dish datasets.
"""