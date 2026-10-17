#!/usr/bin/env python3
"""
curate_data.py
==============
Step 8 of the pipeline. Curates instance windows into per-user train/test
sets with balanced meal / non-meal instances.

Instance quality filter (classify_instance):
  - at least MIN_CGM_POINTS (96) CGM readings in the 8-hour window
  - MEAL: >= MIN_CARBS (10 g) of carbohydrate within [-30, +60] min of the
    forecast origin (the 48th CGM reading)
  - NON_MEAL: no qualifying meal, and either a bolus >= MIN_BOLUS (0.5 U) in
    that interval or a stable trace (rise < 20 mg/dL and SD < 20 mg/dL)
  - BAD (excluded): too few CGM points, or an unexplained rise > 40 mg/dL with
    < 5 g of logged carbohydrate

Curation strategy:
  - Strict 70/30 chronological split (train on past, test on future)
  - 8-hour washout between last train and first test instance
  - 50/50 meal/non-meal balance target within each split
  - Minimum 4-hour gap between selected instances (reduces overlap)
  - Fill to 150 instances with any valid instances if needed

Outputs, in $T1DEXI_OUTPUT_DIR/data_filtering/non_exercise_8h/:
  per_user/user_<id>.pkl         {user_id, train_ids, test_ids}
  train_instances.pkl            all users' train instance ids (flat list)
  test_instances.pkl             all users' test instance ids (flat list)
  filtered_user_instances.pkl    {user_id: train_ids + test_ids}

Usage:
  python curate_data.py
  python curate_data.py --combine-only   # rebuild the combined lists from per_user/
"""
import pickle
import pandas as pd
import os
import argparse
import numpy as np
import ray

from config import OUTPUT_DATA_PATH, FILTERING_OUTPUT_PATH, INSTANCE_FOLDER, num_workers

BASE_DATA_DIR       = OUTPUT_DATA_PATH
FILTERED_OUTPUT_DIR = FILTERING_OUTPUT_PATH
TRAIN_SIZE          = 150
TEST_SIZE           = 150
MIN_CGM_POINTS      = 96
UNEXPLAINED_RISE_TH = 40.0
MEAL_WINDOW_MIN     = -30
MEAL_WINDOW_MAX     =  60
MIN_CARBS           = 10.0
MIN_BOLUS           = 0.5


def load_extracted_features(data_dir):
    """Load the instance table and locate the per-user feature folder."""
    print(f"Loading extracted features from: {data_dir}")
    instances_path = os.path.join(data_dir, 'non_exercise.csv')
    if not os.path.exists(instances_path):
        return {}
    non_ex = pd.read_csv(instances_path)
    return {
        'non_exercise': non_ex,
        'by_user_dir':  os.path.join(data_dir, 'by_user')
    }


def load_user_features(user_id, by_user_dir):
    """Load pre-split per-user parquet files."""
    features = {}
    for name in ['cgm', 'meals', 'bolus', 'basal']:
        fp = os.path.join(by_user_dir, name, f'{user_id}.parquet')
        if os.path.exists(fp):
            df = pd.read_parquet(fp)
            features[name] = {
                eid: grp.reset_index(drop=True)
                for eid, grp in df.groupby('ex')
            }
        else:
            features[name] = {}
    return features


def classify_instance(eid, features):
    """Label one instance MEAL, NON_MEAL or BAD (see module docstring)."""
    cgm   = features.get('cgm',   {}).get(eid, pd.DataFrame())
    meals = features.get('meals', {}).get(eid, pd.DataFrame())
    bolus = features.get('bolus', {}).get(eid, pd.DataFrame())

    if cgm.empty or len(cgm) < MIN_CGM_POINTS:
        return 'BAD'
    time_col = next(
        (c for c in ['timestamp', 'datestamp', 'time'] if c in cgm.columns), None)
    if not time_col:
        return 'BAD'
    cgm[time_col] = pd.to_datetime(cgm[time_col], format='mixed')
    cgm = cgm.sort_values(time_col)
    t0         = cgm.iloc[47][time_col]
    future_cgm = cgm[cgm[time_col] >= t0]['cgm'].values
    if len(future_cgm) < 48:
        return 'BAD'
    rise = np.max(future_cgm) - future_cgm[0]
    total_carbs = 0
    if not meals.empty:
        m_col = next(
            (c for c in ['timestamp', 'datestamp', 'time'] if c in meals.columns), None)
        if m_col:
            meals[m_col] = pd.to_datetime(meals[m_col])
            meals['rel_min'] = (meals[m_col] - t0).dt.total_seconds() / 60.0
            relevant = meals[
                (meals['rel_min'] >= MEAL_WINDOW_MIN) &
                (meals['rel_min'] <= MEAL_WINDOW_MAX)]
            total_carbs = (
                relevant['meal_dose'].sum() if 'meal_dose' in meals.columns else 0)
    if rise > UNEXPLAINED_RISE_TH and total_carbs < 5.0:
        return 'BAD'
    if total_carbs >= MIN_CARBS:
        return 'MEAL'
    total_bolus = 0
    if not bolus.empty:
        b_col = next(
            (c for c in ['timestamp', 'datestamp', 'time'] if c in bolus.columns), None)
        if b_col:
            bolus[b_col] = pd.to_datetime(bolus[b_col])
            bolus['rel_min'] = (bolus[b_col] - t0).dt.total_seconds() / 60.0
            relevant_b = bolus[
                (bolus['rel_min'] >= MEAL_WINDOW_MIN) &
                (bolus['rel_min'] <= MEAL_WINDOW_MAX)]
            total_bolus = (
                relevant_b['bolus'].sum() if 'bolus' in relevant_b.columns else 0)
    if total_bolus >= MIN_BOLUS or (rise < 20.0 and np.std(future_cgm) < 20.0):
        return 'NON_MEAL'
    return 'BAD'


def build_label_cache(user_eids, features):
    """Classify every instance of a user once: {ex: label}."""
    cache = {}
    for eid in user_eids:
        cache[eid] = classify_instance(eid, features)
    return cache


def select_balanced_subset_cached(pool_df, label_cache, target_count,
                                   min_gap_hours=4.0):
    """
    Select balanced temporal subset with minimum gap between instances.
    Enforces min_gap_hours spacing first, then fills remaining slots.
    """
    if pool_df.empty:
        return []

    candidates = []
    for eid in pool_df['ex']:
        label = label_cache.get(eid, 'BAD')
        if label != 'BAD':
            candidates.append({
                'ex':   eid,
                'type': label,
                'time': pool_df[pool_df['ex'] == eid]['time_series_start'].iloc[0]
            })
    if not candidates:
        return []

    cand_df = pd.DataFrame(candidates).sort_values('time')
    cand_df['time'] = pd.to_datetime(cand_df['time'])
    n_meals_target = int(target_count * 0.5)

    def greedy_spaced(rows, n, gap_hours):
        """Pick up to n rows in time order, each at least gap_hours after the previous pick."""
        selected = []
        last_time = None
        for _, row in rows.iterrows():
            if last_time is None or \
               (row['time'] - last_time).total_seconds() / 3600 >= gap_hours:
                selected.append(row['ex'])
                last_time = row['time']
                if len(selected) >= n:
                    break
        return selected

    meals_df     = cand_df[cand_df['type'] == 'MEAL']
    non_meals_df = cand_df[cand_df['type'] == 'NON_MEAL']

    meal_selected     = greedy_spaced(meals_df,     n_meals_target, min_gap_hours)
    remaining_needed  = target_count - len(meal_selected)
    non_meal_selected = greedy_spaced(non_meals_df, remaining_needed, min_gap_hours)
    selected = meal_selected + non_meal_selected

    # Fill remaining with smaller gap if needed
    if len(selected) < target_count:
        selected_set = set(selected)
        for _, row in cand_df.iterrows():
            if len(selected) >= target_count:
                break
            if row['ex'] not in selected_set:
                selected.append(row['ex'])
                selected_set.add(row['ex'])

    return selected[:target_count]


def fill_to_target(selected, pool_df, cache, target):
    """Fill selected list to target count with any non-BAD instance."""
    if len(selected) >= target:
        return selected
    selected_set = set(selected)
    for eid in pool_df['ex']:
        if len(selected) >= target:
            break
        if eid not in selected_set and cache.get(eid, 'BAD') != 'BAD':
            selected.append(eid)
            selected_set.add(eid)
    return selected


@ray.remote
def curate_user(user_id, user_df_dict, by_user_dir, out_dir):
    """Split, filter and select one user's instances; writes per_user/user_<id>.pkl."""
    features = load_user_features(user_id, by_user_dir)

    user_df = pd.DataFrame(user_df_dict)
    user_df['ex'] = user_df['ex'].astype(int)
    user_df['time_series_start'] = pd.to_datetime(user_df['time_series_start'])
    user_df = user_df.sort_values('time_series_start')

    # FILTER TO INSTANCES WITH CGM DATA BEFORE SPLITTING
    cgm_features = features.get('cgm', {})
    valid_ex = set(cgm_features.keys())
    user_df = user_df[user_df['ex'].isin(valid_ex)]
    if len(user_df) == 0:
        print(f"  User {user_id}: no valid instances, skipping")
        result = {'user_id': user_id, 'train_ids': [], 'test_ids': []}
        per_user_dir = os.path.join(out_dir, "per_user")
        os.makedirs(per_user_dir, exist_ok=True)
        with open(os.path.join(per_user_dir, f"user_{user_id}.pkl"), 'wb') as f:
            pickle.dump(result, f)
        return os.path.join(per_user_dir, f"user_{user_id}.pkl")

    # PRE-CLASSIFY ONCE
    all_eids    = user_df['ex'].tolist()
    label_cache = build_label_cache(all_eids, features)

    # STRICT 70/30 CHRONOLOGICAL SPLIT
    split_idx    = int(len(user_df) * 0.7)
    u_train_pool = user_df.iloc[:split_idx]
    u_test_pool  = user_df.iloc[split_idx:]

    # ENFORCE 8H WASHOUT BETWEEN TRAIN AND TEST
    if not u_train_pool.empty and not u_test_pool.empty:
        last_train_time = pd.to_datetime(u_train_pool['time_series_start'].max())
        washout_cutoff  = last_train_time + pd.Timedelta(hours=8)
        u_test_pool = u_test_pool[
            pd.to_datetime(u_test_pool['time_series_start']) >= washout_cutoff
        ]

    # SELECT BALANCED SUBSETS
    train_ids = select_balanced_subset_cached(u_train_pool, label_cache, TRAIN_SIZE)
    test_ids  = select_balanced_subset_cached(u_test_pool,  label_cache, TEST_SIZE)

    # FILL TO 150 WITH ANY VALID INSTANCES
    train_ids = fill_to_target(train_ids, u_train_pool, label_cache, TRAIN_SIZE)
    test_ids  = fill_to_target(test_ids,  u_test_pool,  label_cache, TEST_SIZE)

    n_test_meals = sum(1 for e in u_test_pool['ex'] if label_cache.get(e) == 'MEAL')
    print(f"  User {user_id}: split=0.70  "
          f"test meals={n_test_meals}  train={len(train_ids)}  test={len(test_ids)}")

    # SAVE PER-USER PKL
    result       = {'user_id': user_id, 'train_ids': train_ids, 'test_ids': test_ids}
    per_user_dir = os.path.join(out_dir, "per_user")
    os.makedirs(per_user_dir, exist_ok=True)
    pkl_path = os.path.join(per_user_dir, f"user_{user_id}.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(result, f)
    return pkl_path


def combine_and_save(out_dir):
    """Merge per_user/*.pkl into the combined train/test/filtered lists."""
    per_user_dir = os.path.join(out_dir, "per_user")
    pkl_files    = sorted(f for f in os.listdir(per_user_dir) if f.endswith('.pkl'))

    final_train_ids         = []
    final_test_ids          = []
    filtered_user_instances = {}

    for fname in pkl_files:
        with open(os.path.join(per_user_dir, fname), 'rb') as f:
            result = pickle.load(f)
        uid       = result['user_id']
        train_ids = result['train_ids']
        test_ids  = result['test_ids']
        final_train_ids.extend(train_ids)
        final_test_ids.extend(test_ids)
        filtered_user_instances[str(uid)] = train_ids + test_ids

    print(f"\nCombining {len(pkl_files)} per-user results...")
    print(f"  Total train : {len(final_train_ids)}")
    print(f"  Total test  : {len(final_test_ids)}")

    with open(os.path.join(out_dir, "filtered_user_instances.pkl"), 'wb') as f:
        pickle.dump(filtered_user_instances, f)
    with open(os.path.join(out_dir, "train_instances.pkl"), 'wb') as f:
        pickle.dump(final_train_ids, f)
    with open(os.path.join(out_dir, "test_instances.pkl"), 'wb') as f:
        pickle.dump(final_test_ids, f)

    print("Saved: filtered_user_instances.pkl, train_instances.pkl, test_instances.pkl")

    train_per_user = []
    test_per_user  = []
    for fname in pkl_files:
        with open(os.path.join(per_user_dir, fname), 'rb') as f:
            r = pickle.load(f)
        train_per_user.append(len(r['train_ids']))
        test_per_user.append(len(r['test_ids']))

    print("\nDataset distribution:")
    print(f"  Users: {len(pkl_files)}")
    print(f"  Train - min: {np.min(train_per_user)}  mean: {np.mean(train_per_user):.1f}  max: {np.max(train_per_user)}")
    print(f"  Test  - min: {np.min(test_per_user)}  mean: {np.mean(test_per_user):.1f}  max: {np.max(test_per_user)}")
    print(f"  Users with 150 train: {sum(t == 150 for t in train_per_user)}")
    print(f"  Users with 150 test:  {sum(t == 150 for t in test_per_user)}")

    return final_train_ids, final_test_ids


def main():
    parser = argparse.ArgumentParser(description='Curate per-user train/test instances.')
    parser.add_argument('--folder',       type=str, default=INSTANCE_FOLDER,
                        help='Instance folder under processed_data/ (default: %(default)s)')
    parser.add_argument('--num-cpus',     type=int, default=None)
    parser.add_argument('--combine-only', action='store_true',
                        help='Only rebuild the combined lists from existing per_user/ files')
    args = parser.parse_args()

    data_dir = os.path.join(BASE_DATA_DIR, args.folder)
    out_dir  = os.path.join(FILTERED_OUTPUT_DIR, args.folder)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("CURATE DATA")
    print(f"{'='*60}")
    print(f"  Input  : {data_dir}")
    print(f"  Output : {out_dir}")
    print("  Split  : 70/30 chronological + 8h washout")
    print(f"  Target : {TRAIN_SIZE} train / {TEST_SIZE} test per user")

    if args.combine_only:
        combine_and_save(out_dir)
        return

    features = load_extracted_features(data_dir)
    by_user_dir = os.path.join(data_dir, 'by_user')
    if not features or not os.path.exists(os.path.join(by_user_dir, 'cgm')):
        raise SystemExit(f"[ERROR] non_exercise.csv or by_user/cgm missing in {data_dir}.")

    all_insts = features['non_exercise'].copy()
    all_insts['time_series_start'] = pd.to_datetime(all_insts['time_series_start'])
    unique_users = all_insts['id'].unique()
    print(f"  Users found: {len(unique_users)}")

    if not ray.is_initialized():
        ray.init(num_cpus=args.num_cpus or num_workers(), ignore_reinit_error=True)

    by_user_dir_ref = ray.put(by_user_dir)

    futures = []
    for user_id in unique_users:
        user_df = all_insts[all_insts['id'] == user_id].copy()
        futures.append(
            curate_user.remote(
                user_id, user_df.to_dict('list'), by_user_dir_ref, out_dir))

    print(f"  Launched {len(futures)} workers...")
    pkl_paths = ray.get(futures)
    print(f"  All {len(pkl_paths)} workers done.")

    if ray.is_initialized():
        ray.shutdown()

    train_ids, test_ids = combine_and_save(out_dir)
    print(f"\nCuration complete. Total instances: {len(train_ids) + len(test_ids)}")
    print(f"   Saved to: {out_dir}")


if __name__ == "__main__":
    main()
