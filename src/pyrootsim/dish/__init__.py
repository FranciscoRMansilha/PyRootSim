"""
pyrootsim.dish — Petri dish composition and visualisation.

This subpackage composes multiple individually generated root systems
onto a shared canvas that simulates a petri-dish image. The default
layout was designed to approximate the format used by NPEC's HADES
high-throughput root phenotyping system, which images up to 5
*Arabidopsis thaliana* seedlings per dish. In a future version, the
canvas geometry and placement parameters will be wrapped in a
configuration class so that users can easily adapt the composer to
different experimental setups.

- ``composer``:    Composes synthetic petri-dish images from RSA output.
- ``visualizer``:  Colour-coded rendering and grid visualisation.
"""