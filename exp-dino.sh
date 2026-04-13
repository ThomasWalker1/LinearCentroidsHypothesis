#!/usr/bin/env bash

set -e

python dino/extraction.py
python dino/training.py
python dino/evaluate_dictionaries.py # recreates fig 8c
python dino/evaluate_with_probes.py # recreates fig 9a
python dino/evaluate_firing_distribution.py # recreates fig 9b
