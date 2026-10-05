#!/usr/bin/env python3
"""Script to download and cache all datasets registered in data/registry.yaml."""

from __future__ import annotations

import sys
from acbs.data.loader import load_dataset, load_registry
from acbs.utils.logging import get_logger

logger = get_logger("download_datasets")


def main() -> None:
    registry = load_registry()
    print(f"Found {len(registry)} datasets in registry. Downloading and caching...")

    successful = 0
    skipped = 0
    failed = 0

    for entry in registry:
        name = entry["name"]
        source = entry["source"]
        if source == "kaggle":
            print(f"[-] Skipping Kaggle dataset '{name}' (download manually to data/raw/).")
            skipped += 1
            continue

        try:
            print(f"[*] Loading dataset '{name}' ({source})...", end=" ")
            ds = load_dataset(name)
            print(f"OK ({ds.X.shape[0]} rows, {ds.X.shape[1]} features)")
            successful += 1
        except Exception as exc:
            print(f"FAILED: {exc}")
            failed += 1

    print(f"\nCompleted: {successful} downloaded/cached, {skipped} skipped, {failed} failed.")


if __name__ == "__main__":
    main()
