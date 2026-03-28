#!/usr/bin/env bash
# Quick setup and test runner for Mobile LLM Benchmark Suite
set -e
cd "$(dirname "$0")/.."

echo "=== Installing dependencies ==="
pip install -r requirements.txt -q

echo "=== Running tests ==="
python -m pytest tests/ -q --tb=short

echo "=== Running demo (mock mode) ==="
MOCK_MODE=true python scripts/demo.py

echo "=== All done ==="
