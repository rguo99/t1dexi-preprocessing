"""
process_insulin.py
==================
Step 3 of the pipeline. Converts raw basal insulin records into a 5-minute
grid of units delivered per slot for every participant
($T1DEXI_OUTPUT_DIR/cleaned_data/basal.parquet).

Adapts AID chunking logic from:
    netIOB (CeADAR; Lewis et al., preprint), Apache License 2.0
    (see netiob/NOTICE)

Extends to non-AID devices and adds:
    - duration=0 edge case handling
    - NaN for true data gaps (>7 min)
    - 0 for suspend events
    - Outputs a clean 5-min basal.parquet

Usage:
    python process_insulin.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RAW_DATA_PATH, CLEANED_DATA_PATH, num_workers, require_raw_path
from netiob.base_utils import T1DEXI_Utils
from netiob.insulin_processing import (
    chunk_procedure,
    remove_oddities,
)

utils = None


def fix_duration_zero(df):
    """
    Fix records with FADUR = 00:00:00.
    Handles records where FADUR=00:00:00 which actually means the record
    ran until the next timestamp - potentially multiple days.
    """
    df = df.copy().reset_index(drop=True)
    
    for i in range(len(df)):
        if df.at[i, 'FADUR'] == pd.Timedelta(0):
            if i + 1 < len(df):
                recalc = (df.at[i+1, 'FADTC'] - df.at[i, 'FADTC']).total_seconds() / 3600
                df.at[i, 'FADUR'] = pd.Timedelta(hours=recalc)
            else:
                # Last record with duration 0 - assume 24 hours
                df.at[i, 'FADUR'] = pd.Timedelta(hours=24)
    
    return df


def chunk_non_aid(df):
    """
    Chunk non-AID insulin records into 5-minute intervals.
    Non-AID devices don't have automated adjustments but can still have
    extended boluses (square/dual-square wave) that need chunking.
    """
    # Fix duration=0 edge cases first
    df = fix_duration_zero(df)

    # Handle suspend -> 0
    df.loc[(df['INSSTYPE'] == 'suspend') & df['FASTRESN'].isna(), 'FASTRESN'] = 0

    # Chunk extended basal records (>6 min duration)
    df = chunk_procedure(df, ins_type='BASAL', device_type='TANDEM')

    # Chunk extended bolus records (square/dual-square wave)
    df = chunk_procedure(df, ins_type='BOLUS', device_type='TANDEM')

    return df


def expand_to_5min_grid(df, user_id, cgm_start, cgm_end, is_mdi=False):
    """Expand one user's basal records onto a 5-minute grid between the first and
    last CGM reading; returns timestamp, user_id, basal_units (U per slot)."""
    if df.empty:
        return pd.DataFrame({'timestamp': pd.date_range(cgm_start, cgm_end, freq='5min'),
                            'user_id': user_id, 'basal_units': np.nan})

    if is_mdi:
        ref = df[(df['FACAT'] == 'BASAL') & (df['FATEST'] == 'BASAL INSULIN')].copy()
    else:
        ref = df[(df['FACAT'] == 'BASAL') & (df['FATEST'] == 'BASAL FLOW RATE')].copy()

    ref = ref.sort_values('FADTC').reset_index(drop=True)
    ref['FADTC'] = pd.to_datetime(ref['FADTC'])

    if ref.empty:
        return pd.DataFrame({'timestamp': pd.date_range(cgm_start, cgm_end, freq='5min'),
                            'user_id': user_id, 'basal_units': np.nan})

    effective_start = max(cgm_start, ref['FADTC'].min()).floor('5min')
    effective_end = min(cgm_end, ref['FADTC'].max()).ceil('5min')

    grid = pd.DataFrame({
        'timestamp': pd.date_range(start=effective_start, end=effective_end, freq='5min')
    })
    grid['user_id'] = user_id
    grid['basal_units'] = np.nan

    if is_mdi:
        ref['timestamp'] = ref['FADTC'].dt.round('5min')
        mdi_events = ref.groupby('timestamp')['FASTRESN'].sum().reset_index()
        mdi_events.columns = ['timestamp', 'basal_units']
        grid = grid.merge(mdi_events, on='timestamp', how='left', suffixes=('', '_mdi'))
        grid['basal_units'] = grid['basal_units_mdi']
        grid = grid[['timestamp', 'user_id', 'basal_units']]
    else:
        merged = pd.merge_asof(
            grid,
            ref[['FADTC', 'FASTRESN', 'FADUR', 'INSSTYPE']],
            left_on='timestamp',
            right_on='FADTC',
            direction='backward'
        )
        # A slot is NaN only if outside duration AND gap > 7 min
        # This handles cases where duration < time to next record
        # (pump continues delivering at same rate until next rate change)
        merged['record_end'] = merged['FADTC'] + merged['FADUR']
        merged['within_duration'] = merged['timestamp'] <= merged['record_end']
        merged['gap_min'] = (merged['timestamp'] - merged['FADTC']).dt.total_seconds() / 60
        merged['basal_units'] = np.where(
            merged['INSSTYPE'].str.contains('suspend', na=False),
            0.0,
            np.where(
                (~merged['within_duration']) & (merged['gap_min'] > 7),
                np.nan,
                merged['FASTRESN'] / 12.0
            )
        )
        grid = merged[['timestamp', 'user_id', 'basal_units']]

    return grid

def process_user(args):
    """Process basal insulin for a single user."""
    user_id, user_insulin, cgm_start, cgm_end, device = args

    try:
        if device == 'TANDEM T:SLIM X2 WITH CONTROL IQ':
            user_insulin.loc[(user_insulin['INSSTYPE'] == 'suspend') &
                           user_insulin['FASTRESN'].isna(), 'FASTRESN'] = 0
            chunked = chunk_procedure(user_insulin, ins_type='BASAL', device_type='TANDEM')
            chunked = chunk_procedure(chunked, ins_type='BOLUS', device_type='TANDEM')

        elif device == 'MEDTRONIC 670G IN AUTO MODE':
            user_insulin.loc[(user_insulin['INSSTYPE'] == 'suspend') &
                           user_insulin['FASTRESN'].isna(), 'FASTRESN'] = 0
            chunked = chunk_procedure(user_insulin, ins_type='BASAL', device_type='670G')
            chunked = chunk_procedure(chunked, ins_type='BOLUS', device_type='670G')

        elif device == 'MEDTRONIC 770G IN AUTO MODE':
            # 770G already has correct FADUR and BASAL FLOW RATE records
            # No chunking needed - durations are pre-recorded correctly
            chunked = user_insulin

        else:
            chunked = chunk_non_aid(user_insulin)

        MDI_DEVICE = 'MULTIPLE DAILY INJECTIONS'

        return expand_to_5min_grid(
            chunked[chunked['FACAT'] == 'BASAL'],
            user_id, cgm_start, cgm_end,
            is_mdi=(device == MDI_DEVICE)
        )
    except Exception as e:
        print(f'  Warning: failed for user {user_id}: {e}')
        return pd.DataFrame({'timestamp': pd.date_range(cgm_start, cgm_end, freq='5min'),
                            'user_id': user_id, 'basal_units': np.nan})


def process_basal(insulin_data, cgm_data):
    """Process basal insulin for all users into 5-min grid using multiprocessing."""
    print('\nProcessing basal insulin...')

    # Pre-group by user - much faster than filtering in a loop
    insulin_by_user = dict(list(insulin_data.groupby('USUBJID')))
    cgm_by_user = dict(list(cgm_data.groupby('USUBJID')))

    user_args = []
    for user_id, user_insulin in insulin_by_user.items():
        if user_id not in cgm_by_user:
            continue
        user_cgm = cgm_by_user[user_id]
        cgm_start = pd.to_datetime(user_cgm['LBDTC'].min())
        cgm_end = pd.to_datetime(user_cgm['LBDTC'].max())
        device = user_insulin['DXTRT'].iloc[0] if not user_insulin.empty else None
        user_args.append((user_id, user_insulin, cgm_start, cgm_end, device))

    from multiprocessing import Pool
    from tqdm import tqdm
    n_cpus = num_workers()
    print(f'  Using {n_cpus} CPUs for {len(user_args)} users...')

    with Pool(processes=n_cpus) as pool:
        results = list(tqdm(
            pool.imap(process_user, user_args),
            total=len(user_args),
            desc='  Processing users'
        ))

    basal_df = pd.concat([r for r in results if r is not None], ignore_index=True)
    return basal_df

def validate_basal(basal_df, insulin_data):
    """Sanity check basal output against raw data."""
    print('\nValidating basal output...')
    
    device_map = insulin_data.drop_duplicates('USUBJID').set_index('USUBJID')['DXTRT']
    basal_df['device'] = basal_df['user_id'].map(device_map)

    devices_to_check = [
        'TANDEM T:SLIM X2 WITH CONTROL IQ',
        'MEDTRONIC 670G IN AUTO MODE',
        'MEDTRONIC 770G IN AUTO MODE',
        'INSULET OMNIPOD INSULIN MANAGEMENT SYSTEM',
        'MEDTRONIC 630G',
        'TANDEM T:SLIM X2 WITH BASAL IQ',
    ]

    all_passed = True
    for device in devices_to_check:
        device_users = insulin_data[insulin_data['DXTRT'] == device]['USUBJID'].unique()
        if len(device_users) == 0:
            continue
        user_id = device_users[0]

        user_basal = basal_df[basal_df['user_id'] == user_id].copy()
        user_basal['date'] = pd.to_datetime(user_basal['timestamp']).dt.date
        if user_basal.empty:
            continue

        first_day = user_basal['date'].iloc[1]
        day_basal = user_basal[user_basal['date'] == first_day]
        computed_total = day_basal['basal_units'].sum()

        raw = insulin_data[insulin_data['USUBJID'] == user_id].copy()
        raw['date'] = pd.to_datetime(raw['FADTC']).dt.date
        raw_day = raw[
            (raw['date'] == first_day) &
            (raw['FACAT'] == 'BASAL') &
            (raw['FATEST'] == 'BASAL INSULIN')
        ]
        raw_total = raw_day['FASTRESN'].sum()

        diff = abs(computed_total - raw_total)
        pct = (diff / raw_total * 100) if raw_total > 0 else 0
        passed = pct < 1.0  # allow <1% error
        all_passed = all_passed and passed
        status = 'ok  ' if passed else 'FAIL'
        print(f'  {status} {device[:40]}: {computed_total:.3f} vs {raw_total:.3f} U ({pct:.2f}% error)')

    if all_passed:
        print('All validation checks passed')
    else:
        print('WARNING: some validation checks failed; review the output above')

    basal_df.drop(columns=['device'], inplace=True)

def main():
    global utils
    require_raw_path()
    utils = T1DEXI_Utils(RAW_DATA_PATH + '/')
    os.makedirs(CLEANED_DATA_PATH, exist_ok=True)

    print('=' * 60)
    print('PROCESSING INSULIN DATA')
    print('=' * 60)

    print('\nLoading raw insulin data...')
    raw_insulin = utils._get_data('FACM')[0][[
        'USUBJID', 'FATEST', 'FACAT', 'FADTC',
        'FASTRESN', 'FADUR', 'INSSTYPE', 'INSNMBOL', 'INSEXBOL'
    ]]
    device_data = utils.get_device_data()[0]
    insulin_data = raw_insulin.merge(device_data, on='USUBJID', how='inner')
    insulin_data['FADUR'] = pd.to_timedelta(
        insulin_data['FADUR'].apply(utils.iso8601_duration_to_timedelta)
    )
    insulin_data['FADTC'] = pd.to_datetime(insulin_data['FADTC'])
    insulin_data = insulin_data.sort_values(['USUBJID', 'FADTC'])

    # Cache remove_oddities output to avoid rerunning
    import pickle
    cache_path = os.path.join(CLEANED_DATA_PATH, 'insulin_oddities_removed.pkl')
    if os.path.exists(cache_path):
        print('Loading cached insulin data (skipping remove_oddities)...')
        with open(cache_path, 'rb') as f:
            insulin_data = pickle.load(f)
    else:
        print('Removing oddities...')
        insulin_data = remove_oddities(insulin_data)
        os.makedirs(CLEANED_DATA_PATH, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(insulin_data, f)
        print('Cached insulin data to disk.')

    print('Loading CGM data for study period boundaries...')
    cgm_data = utils._get_data('LB')[0][['USUBJID', 'LBDTC']]
    cgm_data['LBDTC'] = pd.to_datetime(cgm_data['LBDTC'])

    basal_df = process_basal(insulin_data, cgm_data)

    out_path = os.path.join(CLEANED_DATA_PATH, 'basal.parquet')
    basal_df.to_parquet(out_path, index=False)
    print(f'\nSaved basal.parquet: {basal_df.shape[0]:,} rows')
    print(f'  NaN rate: {basal_df["basal_units"].isna().mean():.1%}')
    print(f'  Path: {out_path}')
    print('=' * 60)

    validate_basal(basal_df, insulin_data)


if __name__ == '__main__':
    main()