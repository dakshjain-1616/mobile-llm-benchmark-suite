#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python3 scripts/generate_infographics.py
echo "Assets generated:"
ls -lh assets/*.png 2>/dev/null || echo "No PNGs found"
