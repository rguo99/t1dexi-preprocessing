"""
instance_feature_extraction.py
==============================
Step 6 of the pipeline. Generates the 8-hour non-exercise instance windows
for all users and extracts the CGM, meal, bolus and basal records inside each
window.

Windows advance on a 5-minute stride and are excluded if they overlap an
exercise session or the 2 hours after it. Only users with CGM, bolus, basal
and meal data are included. Instance ids (`ex`) are assigned sequentially
over users in sorted user-id order and, within a user, in time order, so
reruns give identical ids.

Outputs, in $T1DEXI_OUTPUT_DIR/processed_data/non_exercise_8h/:
  non_exercise.csv                 ex, id, time_series_start, time_series_end
  by_user/<feature>/<user>.parquet records inside each window, keyed by ex

Prerequisites: get_netiob_cleaned_data.py, process_insulin.py,
build_carb_lookup.py and split_parquets_by_user.py.

Usage:
    python instance_feature_extraction.py                   # windows + features
    python instance_feature_extraction.py --instances-only  # windows only
"""

import argparse
import os
from datetime import timedelta

import pandas as pd
import ray
from tqdm import tqdm

from config import (
    MEALS_PARQUET, GLUCOSE_PARQUET, EXERCISE_CSV,
    BOLUS_PARQUET, BASAL_PARQUET,
    OUTPUT_DATA_PATH, BY_USER_DIR, INSTANCE_FOLDER,
    validate_paths, num_workers
)


# Helpers

def _load_users_with_all_data():
    """Return the set of user ids that have CGM, bolus, basal and meal data."""
    cgm_users   = set(pd.read_parquet(GLUCOSE_PARQUET)['USUBJID'].astype(str).unique())
    bolus_users = set(pd.read_parquet(BOLUS_PARQUET)['USUBJID'].astype(str).unique())
    meals_df    = pd.read_parquet(MEALS_PARQUET)
    meal_users  = set(meals_df['USUBJID'].astype(str).unique())

    # Basal uses user_id column
    by_user_basal = os.path.join(BY_USER_DIR, 'basal')
    if os.path.exists(by_user_basal) and len(os.listdir(by_user_basal)) > 0:
        basal_users = set(f.replace('.parquet', '') for f in os.listdir(by_user_basal))
    else:
        basal_users = set(pd.read_parquet(BASAL_PARQUET)['user_id'].astype(str).unique())

    return cgm_users & bolus_users & meal_users & basal_users


# Window generation

@ray.remote
def process_user_windows(user_id, cgm_df, exercise_df):
    """Non-exercise 8-hour windows for one user.

    Windows start at the user's first CGM reading and advance in 5-minute
    steps until the user's last reading; a window is kept unless it overlaps
    an exercise session or the 2 hours after one.
    """
    user_cgm      = cgm_df[cgm_df['id'] == user_id].sort_values('datestamp')
    user_exercise = exercise_df[exercise_df['id'] == user_id]

    if len(user_cgm) == 0:
        return user_id, []

    start_time   = user_cgm['datestamp'].min()
    end_time     = user_cgm['datestamp'].max()
    ex_intervals = list(zip(user_exercise['ex_start_time'], user_exercise['ex_end_time']))

    instances = []
    current_start = start_time

    while current_start + timedelta(hours=8) <= end_time:
        current_end = current_start + timedelta(hours=8)
        overlaps = any(
            current_start < ex_end + timedelta(hours=2) and current_end > ex_start
            for ex_start, ex_end in ex_intervals
        )
        if not overlaps:
            instances.append({
                'ex': None, 'id': user_id,
                'time_series_start': current_start,
                'time_series_end':   current_end
            })
        current_start += timedelta(minutes=5)

    return user_id, instances


def _generate_windows(users, cgm_ref, exercise_ref, out_path, filename):
    """Generate windows for every user with Ray, assign global instance ids
    and save them as <out_path>/<filename> (columns ex, id,
    time_series_start, time_series_end)."""
    futures = [
        process_user_windows.remote(uid, cgm_ref, exercise_ref)
        for uid in sorted(users)
    ]

    remaining = futures.copy()
    with tqdm(total=len(futures), desc='  Generating windows') as pbar:
        while remaining:
            done, remaining = ray.wait(remaining, num_returns=1)
            pbar.update(len(done))
    # Collect results in sorted user-id order, not completion order, so
    # that instance ids are deterministic.
    results = ray.get(futures)

    all_instances = []
    global_ex_id = 0
    for _, user_instances in results:
        for inst in user_instances:
            global_ex_id += 1
            inst['ex'] = global_ex_id
            all_instances.append(inst)

    df = pd.DataFrame(all_instances)
    os.makedirs(out_path, exist_ok=True)
    df.to_csv(os.path.join(out_path, filename), index=False)
    print(f"[*] Saved {len(df):,} instances -> {out_path}/{filename}")
    return df


# Main

def get_non_exercise_windows():
    """Generate and save the non-exercise windows; returns the instance table."""
    print(f"\n[*] Non-exercise windows -> {INSTANCE_FOLDER}")

    if not ray.is_initialized():
        ray.init(num_cpus=num_workers(), ignore_reinit_error=True)

    exercise_data = pd.read_csv(EXERCISE_CSV)
    exercise_data['ex_start_time'] = pd.to_datetime(exercise_data['ex_start_time'])
    exercise_data['ex_end_time']   = pd.to_datetime(exercise_data['ex_end_time'])

    cgm = pd.read_parquet(GLUCOSE_PARQUET)
    cgm.rename(columns={'USUBJID': 'id', 'LBDTC': 'datestamp'}, inplace=True)
    cgm['datestamp'] = pd.to_datetime(cgm['datestamp'])

    users = _load_users_with_all_data()

    cgm           = cgm[cgm['id'].isin(users)]
    exercise_data = exercise_data[exercise_data['id'].isin(users)]

    cgm_ref      = ray.put(cgm)
    exercise_ref = ray.put(exercise_data)

    out_path = os.path.join(OUTPUT_DATA_PATH, INSTANCE_FOLDER)
    return _generate_windows(users, cgm_ref, exercise_ref, out_path, 'non_exercise.csv')


def main():
    parser = argparse.ArgumentParser(
        description='Generate 8-hour non-exercise instance windows and extract features.')
    parser.add_argument('--instances-only', action='store_true',
                        help='Only generate windows, skip feature extraction')
    args = parser.parse_args()

    get_non_exercise_windows()
    if ray.is_initialized():
        ray.shutdown()
    if not args.instances_only:
        from ultra_fast_features_per_user import extract_features_per_user
        extract_features_per_user(INSTANCE_FOLDER, ['cgm', 'meals', 'basal', 'bolus'],
                                  'non_exercise.csv')


if __name__ == '__main__':
    if validate_paths():
        main()
    else:
        raise SystemExit(1)
    if ray.is_initialized():
        ray.shutdown()
