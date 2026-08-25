#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"/src
python cp_milp_exact_baselines.py --mode both --time-limit 300 "$@"
