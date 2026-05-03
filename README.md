# The Linear Centroids Hypothesis

This is the official repository for [The Linear Centroids Hypothesis: How Deep Networks Represent Features.]()

In `centroids.py` are classes for computing centroids and local centroids for PyTorch modules.

`exp-inr.ipynb` explores the implicit neural representation example of fig 1.

`exp-spurious_correlations.py` recreates third panel of fig 2.

The results of Section 4.1 can be recreated by running `exp-dino.sh`. (Note this installs the Imagenette dataset through torchvision).

The results of Section 4.2 can be recreated by running `exp-circuit_discovery.py`.

The results of Section 4.3 can be recreated by running `exp-probes.sh`. (This requires downloading the `datasets` directory from [here](https://github.com/saprmarks/geometry-of-truth) and placing it into `outputs/probes`).

The results of Section 4.4 can be recreated by running `exp-local_centroids.py`. (By default this requires [RobustBench](https://github.com/RobustBench/robustbench), although this can be avoided by putting `RB_MODEL=''`).