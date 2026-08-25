#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"/src
python quav_style_baseline.py "$@"
