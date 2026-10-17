"""
Per-user feature extraction for instance windows (numpy broadcasting + Ray).
===========================================================================
For every instance window in <processed_data>/<data_type>/<instances_file>,
collects the CGM, meal, bolus and basal records that fall inside the window and writes one parquet file per
user and feature to <processed_data>/<data_type>/by_user/<feature>/<user>.parquet.

Each user is processed in its own Ray task; within a user, the
timestamp-in-window join is a vectorized numpy broadcast.

Usage (normally called from instance_feature_extraction.py):
  python ultra_fast_features_per_user.py --data-type non_exercise_8h --features cgm meals basal bolus
"""

import argparse
import os
import pandas as pd
import numpy as np
import ray
from config import (
    OUTPUT_DATA_PATH, BY_USER_DIR, CARB_LOOKUP_PARQUET, INSTANCE_FOLDER, num_workers
)


# ============================================================================
# NUMPY BROADCASTING HELPERS
# ============================================================================

def _point_in_window(timestamps, inst_start, inst_end, inst_ex, max_cells=20_000_000):
    """
    Vectorized point-in-window assignment using numpy broadcasting.

    For each (timestamp, instance) pair, checks if timestamp falls within
    [inst_start, inst_end]. Returns aligned (row_indices, inst_ex_values),
    ordered by row and then by window.

    The (N, W) comparison is evaluated in blocks of rows so that at most
    `max_cells` booleans are held at once; the result is identical to a
    single full broadcast.

    Args:
        timestamps : np.ndarray shape (N,) - feature timestamps
        inst_start : np.ndarray shape (W,) - instance start times
        inst_end   : np.ndarray shape (W,) - instance end times
        inst_ex    : np.ndarray shape (W,) - instance ex IDs

    Returns:
        r_idx  : row indices into the feature array
        ex_arr : corresponding ex IDs
    """
    n, w = len(timestamps), len(inst_start)
    block = max(1, max_cells // max(w, 1))
    r_parts, w_parts = [], []
    for lo in range(0, n, block):
        ts = timestamps[lo:lo + block]
        in_window = (
            (ts[:, None] >= inst_start[None, :]) &
            (ts[:, None] <= inst_end[None, :])
        )  # shape (block, W)
        r, wi = np.where(in_window)
        r_parts.append(r + lo)
        w_parts.append(wi)
    if not r_parts:
        return np.array([], dtype=np.int64), inst_ex[:0]
    r_idx = np.concatenate(r_parts)
    w_idx = np.concatenate(w_parts)
    return r_idx, inst_ex[w_idx]


# ============================================================================
# PER-USER EXTRACTION FUNCTIONS (called inside Ray tasks)
# ============================================================================

def _cgm_for_user(user_inst, user_cgm):
    """CGM readings (cgm, datestamp) inside each window of one user."""
    if user_cgm.empty or user_inst.empty:
        return pd.DataFrame()

    cgm_ts     = user_cgm['datestamp'].values
    inst_start = user_inst['time_series_start'].values
    inst_end   = user_inst['time_series_end'].values
    inst_ex    = user_inst['ex'].values

    r_idx, ex_arr = _point_in_window(cgm_ts, inst_start, inst_end, inst_ex)
    if len(r_idx) == 0:
        return pd.DataFrame()

    sub = user_cgm.iloc[r_idx].copy().reset_index(drop=True)
    sub['ex'] = ex_arr
    return sub[['ex', 'cgm', 'datestamp', 'device_type']]


def _meals_for_user(user_inst, user_meals):
    """Meal records (meal_dose, datestamp, MLTRT, MLCAT) inside each window."""
    if user_meals.empty or user_inst.empty:
        return pd.DataFrame()

    meal_ts    = user_meals['datestamp'].values
    inst_start = user_inst['time_series_start'].values
    inst_end   = user_inst['time_series_end'].values
    inst_ex    = user_inst['ex'].values

    r_idx, ex_arr = _point_in_window(meal_ts, inst_start, inst_end, inst_ex)
    if len(r_idx) == 0:
        return pd.DataFrame()

    sub = user_meals.iloc[r_idx].copy().reset_index(drop=True)
    sub['ex'] = ex_arr
    cols = ['ex', 'meal_dose', 'datestamp', 'device_type']
    for c in ['MLTRT', 'MLCAT']:
        if c in sub.columns:
            cols.append(c)
        else:
            sub[c] = ''
            cols.append(c)
    return sub[cols]


def _bolus_for_user(user_inst, user_bolus):
    """Bolus records (bolus, datestamp) inside each window."""
    if user_bolus.empty or user_inst.empty:
        return pd.DataFrame()

    bolus_ts   = user_bolus['datestamp'].values
    inst_start = user_inst['time_series_start'].values
    inst_end   = user_inst['time_series_end'].values
    inst_ex    = user_inst['ex'].values

    r_idx, ex_arr = _point_in_window(bolus_ts, inst_start, inst_end, inst_ex)
    if len(r_idx) == 0:
        return pd.DataFrame()

    sub = user_bolus.iloc[r_idx].copy().reset_index(drop=True)
    sub['ex'] = ex_arr
    return sub[['ex', 'bolus', 'datestamp', 'device_type']]


def _basal_for_user(user_inst, user_basal):
    """5-minute basal slots (timestamp, basal_units) inside each window."""
    if user_basal.empty or user_inst.empty:
        return pd.DataFrame()

    # New basal format: timestamp + basal_units (5-min grid)
    bas_ts     = user_basal['timestamp'].values
    inst_start = user_inst['time_series_start'].values
    inst_end   = user_inst['time_series_end'].values
    inst_ex    = user_inst['ex'].values

    r_idx, ex_arr = _point_in_window(bas_ts, inst_start, inst_end, inst_ex)
    if len(r_idx) == 0:
        return pd.DataFrame()

    sub = user_basal.iloc[r_idx].copy().reset_index(drop=True)
    sub['ex'] = ex_arr
    return sub[['ex', 'timestamp', 'basal_units', 'user_id']]


# ============================================================================
# RAY REMOTE TASK: one task per user
# ============================================================================

# (id, datestamp) -> consumed carbohydrate lookup, loaded lazily once per Ray
# worker process (see the meals block in extract_user()).
_CARB_LOOKUP = None

FEATURES = ('cgm', 'meals', 'bolus', 'basal')


def _load_user_parquet(feature, user_id, by_user_dir):
    """Load by_user/<feature>/<user_id>.parquet (written by split_parquets_by_user.py)."""
    path = os.path.join(by_user_dir, feature, f'{user_id}.parquet')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found; run split_parquets_by_user.py first.')
    return pd.read_parquet(path)


@ray.remote
def extract_user(user_id, user_inst_dict, features_to_extract, by_user_dir, out_dirs):
    """Collect each requested feature's records inside every window of one user
    and write them to <out_dirs[feature]>/<user_id>.parquet.

    Writing inside the task keeps only one user's records in memory per worker.
    Returns {feature: n_rows} for the features that had at least one record.
    """
    user_inst = pd.DataFrame(user_inst_dict)
    user_inst['time_series_start'] = pd.to_datetime(user_inst['time_series_start'])
    user_inst['time_series_end']   = pd.to_datetime(user_inst['time_series_end'])

    results = {}

    def get_user_data(feature, id_col):
        """Load this user's per-user table for one feature."""
        return _load_user_parquet(feature, user_id, by_user_dir)

    if 'cgm' in features_to_extract:
        user_cgm = get_user_data('cgm', 'USUBJID')
        if 'LBORRES' in user_cgm.columns:
            user_cgm.rename(columns={'USUBJID': 'id', 'LBORRES': 'cgm', 'LBDTC': 'datestamp'}, inplace=True)
        user_cgm['datestamp'] = pd.to_datetime(user_cgm.get('datestamp', user_cgm.get('LBDTC')), errors='coerce')
        if 'device_type' not in user_cgm.columns:
            user_cgm['device_type'] = ''
        df = _cgm_for_user(user_inst, user_cgm)
        if not df.empty:
            df['user_id'] = user_id
            results['cgm'] = df.to_dict('list')

    if 'meals' in features_to_extract:
        user_meals = get_user_data('meals', 'USUBJID')
        if 'MLDTC' in user_meals.columns:
            user_meals.rename(columns={'USUBJID': 'id', 'MLDTC': 'datestamp', 'MLDOSE': 'meal_dose'}, inplace=True)
        user_meals['datestamp'] = pd.to_datetime(user_meals['datestamp'], errors='coerce')
        # FAMLPM records each meal as a TAKEN / RETURNED / CONSUMED triple
        # (T1DEXI plate-waste protocol); only CONSUMED is intake. Overwrite
        # meal_dose with the CONSUMED value from carb_consumed_lookup.parquet
        # (built by build_carb_lookup.py from FACAT == 'CONSUMED' rows) and
        # then drop exact duplicates, so each meal event contributes once with
        # its consumed amount, including meals where some food was returned.
        global _CARB_LOOKUP
        if _CARB_LOOKUP is None:
            _CARB_LOOKUP = pd.read_parquet(CARB_LOOKUP_PARQUET)
        user_meals = user_meals.merge(_CARB_LOOKUP, on=['id', 'datestamp'], how='left')
        has_truth = user_meals['consumed_carb_dose'].notna()
        user_meals.loc[has_truth, 'meal_dose'] = user_meals.loc[has_truth, 'consumed_carb_dose']
        user_meals = user_meals.drop(columns=['consumed_carb_dose'])
        dedup_cols = [c for c in ['id', 'datestamp', 'meal_dose', 'MLCAT', 'MLTRT']
                      if c in user_meals.columns]
        user_meals = user_meals.drop_duplicates(subset=dedup_cols, keep='first')
        user_meals = user_meals[user_meals['meal_dose'] != 0]
        if 'device_type' not in user_meals.columns:
            user_meals['device_type'] = ''
        df = _meals_for_user(user_inst, user_meals)
        if not df.empty:
            df['user_id'] = user_id
            results['meals'] = df.to_dict('list')

    if 'bolus' in features_to_extract:
        user_bolus = get_user_data('bolus', 'USUBJID')
        if 'FADTC' in user_bolus.columns:
            user_bolus.rename(columns={'USUBJID': 'id', 'FASTRESN': 'bolus', 'FADTC': 'datestamp'}, inplace=True)
        user_bolus['datestamp'] = pd.to_datetime(user_bolus['datestamp'], errors='coerce')
        if 'device_type' not in user_bolus.columns:
            user_bolus['device_type'] = ''
        df = _bolus_for_user(user_inst, user_bolus)
        if not df.empty:
            df['user_id'] = user_id
            results['bolus'] = df.to_dict('list')

    if 'basal' in features_to_extract:
        user_basal = get_user_data('basal', 'user_id')
        user_basal['timestamp'] = pd.to_datetime(user_basal['timestamp'], errors='coerce')
        user_basal = user_basal.dropna(subset=['basal_units'])
        if 'device_type' not in user_basal.columns:
            user_basal['device_type'] = ''
        df = _basal_for_user(user_inst, user_basal)
        if not df.empty:
            df['user_id'] = user_id
            results['basal'] = df.to_dict('list')

    written = {}
    for feat, data in results.items():
        df = pd.DataFrame(data)
        if df.empty:
            continue
        uid = df['user_id'].iloc[0] if 'user_id' in df.columns else 'unknown'
        df.to_parquet(os.path.join(out_dirs[feat], f'{uid}.parquet'), index=False)
        written[feat] = len(df)
    return written


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def extract_features_per_user(
        data_type=INSTANCE_FOLDER,
        features_to_extract=None,
        instances_file='non_exercise.csv',
        num_cpus=None):

    """Extract features for every window in <processed_data>/<data_type>/<instances_file>
    and write by_user/<feature>/<user>.parquet next to it."""
    if features_to_extract is None:
        features_to_extract = ['cgm', 'meals', 'basal', 'bolus']
    if num_cpus is None:
        num_cpus = num_workers()

    print(f"\n{'='*60}")
    print("PER-USER FEATURE EXTRACTION (numpy broadcast + Ray)")
    print(f"{'='*60}")
    print(f"  data_type : {data_type}")
    print(f"  features  : {features_to_extract}")
    print(f"  num_cpus  : {num_cpus}")

    instances_dir = os.path.join(OUTPUT_DATA_PATH, data_type)
    instances_csv = os.path.join(instances_dir, instances_file)

    if not os.path.exists(instances_csv):
        print(f"ERROR: Instance file not found at: {instances_csv}")
        return {}

    print(f"\nLoading instances from: {instances_csv}")
    instances_df = pd.read_csv(instances_csv)
    instances_df['time_series_start'] = pd.to_datetime(instances_df['time_series_start'])
    instances_df['time_series_end']   = pd.to_datetime(instances_df['time_series_end'])
    unique_users = instances_df['id'].unique()
    print(f"  Found {len(unique_users)} users, {len(instances_df)} instances")

    unknown = set(features_to_extract) - set(FEATURES)
    if unknown:
        raise ValueError(f"Unknown features {sorted(unknown)}; choose from {FEATURES}")
    by_user_dir = BY_USER_DIR
    if not os.path.isdir(by_user_dir):
        raise SystemExit(f"{by_user_dir} not found; run split_parquets_by_user.py first.")

    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus, ignore_reinit_error=True)

    # Per-user output folders (combine_by_user.py writes the combined tables)
    out_dirs = {}
    for feat in features_to_extract:
        out_dirs[feat] = os.path.join(instances_dir, 'by_user', feat)
        os.makedirs(out_dirs[feat], exist_ok=True)

    # LAUNCH ONE RAY TASK PER USER (each task writes its own files)
    print(f"Launching {len(unique_users)} Ray tasks...")
    futures = []
    for user_id in unique_users:
        user_inst = instances_df[instances_df['id'] == user_id].copy()
        futures.append(
            extract_user.remote(
                user_id,
                user_inst.to_dict('list'),
                features_to_extract,
                by_user_dir,
                out_dirs
            )
        )

    saved_counts = {f: 0 for f in features_to_extract}
    for written in ray.get(futures):
        for feat in written:
            saved_counts[feat] += 1
    print(f"  All {len(futures)} tasks done.")

    if ray.is_initialized():
        ray.shutdown()

    for feat, count in saved_counts.items():
        print(f"  {feat}: {count} users saved -> by_user/{feat}/")

    return {}


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-type', type=str, default=INSTANCE_FOLDER)
    parser.add_argument('--features', nargs='+', choices=FEATURES,
                        default=['cgm', 'meals', 'basal', 'bolus'])
    parser.add_argument('--num-cpus', type=int, default=None)
    parser.add_argument('--instances-file', type=str, default='non_exercise.csv')
    args = parser.parse_args()

    extract_features_per_user(
        data_type=args.data_type,
        features_to_extract=args.features,
        instances_file=args.instances_file,
        num_cpus=args.num_cpus
    )