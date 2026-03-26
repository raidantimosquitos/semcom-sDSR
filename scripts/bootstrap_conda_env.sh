#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

conda env create -f environment.yml
conda run -n sDSR pip install --no-build-isolation pyldpc
