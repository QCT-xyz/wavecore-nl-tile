#!/usr/bin/env bash
set -euo pipefail
python examples/plot_xi_vs_j.py
ls -l artifacts || true
