#!/usr/bin/env bash

set -e

python dino/extraction.py
python dino/training.py
python dino/evaluate_dictionaries.py
python dino/evaluate_with_probes.py
python dino/evaluate_firing_distribution.py
