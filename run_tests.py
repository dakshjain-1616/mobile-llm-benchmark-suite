#!/usr/bin/env python3
"""Test runner script — call directly: python run_tests.py"""
import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Install deps
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])

# Run tests
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
    capture_output=False,
)
sys.exit(result.returncode)
