#!/usr/bin/env bash

set -e

python probes/extraction.py --datasets cities cities_cities_conj cities_cities_disj common_claim_true_false companies_true_false counterfact_true_false large_than likely neg_cities neg_sp_en_trans smaller_than sp_en_trans
python probes/generalization.py
