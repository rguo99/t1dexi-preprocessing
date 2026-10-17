"""
build_carb_lookup.py
====================
Step 4 of the pipeline. Builds the consumed-carbohydrate lookup used during
meal feature extraction.

T1DEXI's FAMLPM domain records each meal's carbohydrate total three times
(FACAT = TAKEN, RETURNED, CONSUMED) under the study's photo plate-waste
protocol. Only CONSUMED is the amount eaten. This script keeps the
'Dietary Total Carbohydrate' (FATESTCD == 'DCARBT') rows with
FACAT == 'CONSUMED' and writes one row per (user, timestamp):

    id, datestamp, consumed_carb_dose

to $T1DEXI_OUTPUT_DIR/cleaned_data/carb_consumed_lookup.parquet.

Prerequisite: convert_xpt_to_parquet.py (reads FAMLPM.parquet).

Usage:
    python build_carb_lookup.py
"""

import os

import pandas as pd

from config import RAW_DATA_PATH, CLEANED_DATA_PATH, CARB_LOOKUP_PARQUET, require_raw_path


def main():
    require_raw_path()
    famlpm_path = os.path.join(RAW_DATA_PATH, 'FAMLPM.parquet')
    if not os.path.exists(famlpm_path):
        raise SystemExit(f'{famlpm_path} not found; run convert_xpt_to_parquet.py first.')

    print('=' * 60)
    print('BUILDING CONSUMED-CARBOHYDRATE LOOKUP')
    print('=' * 60)

    df = pd.read_parquet(famlpm_path)
    carbs = df[(df['FATESTCD'] == 'DCARBT') & (df['FACAT'] == 'CONSUMED')].copy()
    lookup = carbs[['USUBJID', 'FADTC', 'FASTRESN']].rename(
        columns={'USUBJID': 'id', 'FADTC': 'datestamp', 'FASTRESN': 'consumed_carb_dose'})
    lookup['id'] = lookup['id'].astype(str)
    lookup['datestamp'] = pd.to_datetime(lookup['datestamp'], errors='coerce')

    n_dup = int(lookup.duplicated(subset=['id', 'datestamp']).sum())
    lookup = lookup.drop_duplicates(subset=['id', 'datestamp'], keep='first')

    os.makedirs(CLEANED_DATA_PATH, exist_ok=True)
    lookup.to_parquet(CARB_LOOKUP_PARQUET, index=False)
    print(f'  CONSUMED rows: {len(carbs):,}  duplicate (id, datestamp) dropped: {n_dup}')
    print(f'  Saved {len(lookup):,} rows -> {CARB_LOOKUP_PARQUET}')
    print('=' * 60)


if __name__ == '__main__':
    main()
