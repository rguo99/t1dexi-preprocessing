"""
combine_by_user.py
==================
Step 7 of the pipeline. Concatenates the per-user feature files written by
instance_feature_extraction.py,

    processed_data/<folder>/by_user/<feature>/<user>.parquet

into one table per feature, processed_data/<folder>/<feature>.parquet. Users
are concatenated in sorted file-name order. CSV copies are also written for
the meal and bolus tables (pass --csv-all for CGM and basal too; those CSVs
are ~15 GB each).

Usage:
    python combine_by_user.py [--folder non_exercise_8h] [--csv-all]
"""

import argparse
import glob
import os

import pandas as pd

from config import OUTPUT_DATA_PATH, INSTANCE_FOLDER

FEATURES = ['cgm', 'meals', 'bolus', 'basal']
CSV_FEATURES = ['meals', 'bolus']


def main():
    parser = argparse.ArgumentParser(description='Combine per-user feature files.')
    parser.add_argument('--folder', default=INSTANCE_FOLDER)
    parser.add_argument('--csv-all', action='store_true',
                        help='Write CSV copies for every feature, not just meals and bolus.')
    args = parser.parse_args()

    root = os.path.join(OUTPUT_DATA_PATH, args.folder)
    print('=' * 60)
    print(f'COMBINING PER-USER FEATURE FILES in {root}')
    print('=' * 60)
    for feat in FEATURES:
        files = sorted(glob.glob(os.path.join(root, 'by_user', feat, '*.parquet')))
        if not files:
            print(f'  {feat}: no per-user files, skipping')
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df.to_parquet(os.path.join(root, f'{feat}.parquet'), index=False)
        if args.csv_all or feat in CSV_FEATURES:
            df.to_csv(os.path.join(root, f'{feat}.csv'), index=False)
        print(f'  {feat}: {len(files)} users, {len(df):,} rows')
    print('=' * 60)


if __name__ == '__main__':
    main()
