"""
convert_xpt_to_parquet.py
=========================
Step 1 of the pipeline. One-time conversion of the raw T1DEXI .xpt domains
used by the pipeline to parquet.

Usage:
    python convert_xpt_to_parquet.py

Output: parquet files written alongside the .xpt files in T1DEXI_RAW_DIR.
"""

import os
import pyreadstat
from config import RAW_T1DEXI_PATH, require_raw_path

# Only convert files actually used by the pipeline
XPT_FILES = [
    'FACM',    # insulin
    'LB',      # glucose/CGM
    'ML',      # rescue carbs
    'FAMLPM',  # per-meal carbs
    'DX',      # device data
    'PR',      # exercise
    'DM',      # demographics
]

def convert_xpt_to_parquet(name):
    """Convert <name>.xpt to <name>.parquet in T1DEXI_RAW_DIR (skipped if present)."""
    xpt_path     = os.path.join(RAW_T1DEXI_PATH, f'{name}.xpt')
    parquet_path = os.path.join(RAW_T1DEXI_PATH, f'{name}.parquet')

    if os.path.exists(parquet_path):
        print(f'  {name}.parquet already exists, skipping')
        return

    if not os.path.exists(xpt_path):
        print(f'  WARNING: {name}.xpt not found, skipping')
        return

    print(f'  Converting {name}.xpt...', end=' ', flush=True)

    df, _ = pyreadstat.read_xport(xpt_path)

    df.to_parquet(parquet_path, index=False)
    size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    print(f'done ({size_mb:.1f} MB)')

def main():
    require_raw_path()

    print('=' * 60)
    print('CONVERTING T1DEXI .XPT FILES TO PARQUET')
    print('=' * 60)
    print(f'\nSource: {RAW_T1DEXI_PATH}\n')

    for name in XPT_FILES:
        convert_xpt_to_parquet(name)

    print('\nDone.')
    print('=' * 60)


if __name__ == '__main__':
    main()