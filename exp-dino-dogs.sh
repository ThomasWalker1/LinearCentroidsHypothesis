#!/usr/bin/env bash

set -e

python dino-dogs/extraction.py
python dino/training.py
python dino/evaluate_with_probes.py
