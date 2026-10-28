"""Shared bootstrap utilities for AutoML worker scripts.

Each framework worker script is a standalone Python file that:
  1. Reads a TrainSpec JSON from stdin.
  2. Fits the framework on X_train, y_train.
  3. Predicts probabilities on X_test.
  4. Writes a single RunResult JSON line to stdout.

Workers are designed to run inside isolated venvs and have NO imports
from the host acbs package — they only depend on stdlib + their framework.
"""
# This file is intentionally minimal; actual logic lives in per-framework scripts.
