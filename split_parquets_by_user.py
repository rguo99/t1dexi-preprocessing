"""
split_parquets_by_user.py
=========================
Step 5 of the pipeline. Splits the cleaned parquet files by user into
$T1DEXI_OUTPUT_DIR/by_user/<feature>/<user>.parquet for fast per-user loading
during feature extraction. A feature folder that already contains files is
skipped.

Usage:
    python split_parquets_by_user.py
"""

import os
import pandas as pd
from multiprocessing import Pool
from config import (
    GLUCOSE_PARQUET, MEALS_PARQUET, BOLUS_PARQUET,
    BASAL_PARQUET, BY_USER_DIR, num_workers,
)

FILES = {
    'cgm':        (GLUCOSE_PARQUET, 'USUBJID'),
    'meals':      (MEALS_PARQUET,   'USUBJID'),
    'bolus':      (BOLUS_PARQUET,   'USUBJID'),
    'basal':      (BASAL_PARQUET,   'user_id'),
}


def _save_user(args):
    """Write one user's rows to <out_dir>/<user_id>.parquet."""
    user_id, user_df, out_dir = args
    user_df.to_parquet(os.path.join(out_dir, f'{user_id}.parquet'), index=False)


def split_parquet_by_user(name, path, id_col):
    """Split one cleaned table into by_user/<name>/<user>.parquet."""
    print(f'\nSplitting {name}...')
    out_dir = os.path.join(BY_USER_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    existing = len(os.listdir(out_dir))
    if existing > 0:
        print(f'  Already split ({existing} files), skipping')
        return

    print(f'  Loading {os.path.basename(path)}...', flush=True)
    df = pd.read_parquet(path)
    print(f'  {len(df):,} rows - grouping by user...', flush=True)

    # groupby is O(n) - much faster than filtering per user
    grouped = df.groupby(id_col)
    user_args = [(uid, grp.copy(), out_dir) for uid, grp in grouped]
    n_users = len(user_args)
    n_cpus = num_workers()
    print(f'  {n_users} users, saving with {n_cpus} CPUs...', flush=True)

    with Pool(processes=n_cpus) as pool:
        for i, _ in enumerate(pool.imap_unordered(_save_user, user_args)):
            if (i + 1) % 100 == 0 or (i + 1) == n_users:
                print(f'  {i+1}/{n_users} saved...', flush=True)

    print(f'  Done -> {out_dir}/')


def main():
    print('=' * 60)
    print('SPLITTING PARQUETS BY USER')
    print('=' * 60)
    print(f'Output: {BY_USER_DIR}\n')

    for name, (path, id_col) in FILES.items():
        split_parquet_by_user(name, path, id_col)

    print('\nAll done.')
    print('=' * 60)


if __name__ == '__main__':
    main()
