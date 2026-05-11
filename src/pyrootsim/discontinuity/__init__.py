"""
Discontinuity injection for synthetic root system masks.

This subpackage creates controlled occlusions (cuts) in composed petri dish
masks, producing 2, 3, or 4 connected components within oriented bounding
boxes (OBBs). These simulate real-world segmentation challenges where roots
may be partially obscured or interrupted.

Modules
-------
build_roots
    Vectorised root data builder. Called once per dish to construct skeleton
    graphs, geodesic distance maps, and junction metadata for all roots.
cut_2cc
    Two-connected-component cuts: clean half-plane cuts on primary or lateral
    root mid-bodies, lateral-base cuts near junctions, and dedicated top-tip
    cuts.
cut_3cc
    Three-connected-component cuts: junction-targeting cuts on primaries with
    optional lateral extension, plus dedicated top-tip junction cuts.
cut_4cc
    Four-connected-component cuts: X-crossing cuts at overlap regions between
    two roots, and bilateral junction cuts where a primary meets two laterals.
orchestrator
    Full pipeline that sequences 2cc → 3cc → 4cc on each dish with a shared
    tracker and mask, applies post-hoc validation, and supports
    multiprocessing.
--------
TO ADD IN THE FUTURE: a module for random occlusions without a specific topology
"""