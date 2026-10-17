"""
Path configuration for the T1DEXI preprocessing pipeline.

Two environment variables control where data is read from and written to:

  T1DEXI_RAW_DIR     folder containing the raw T1DEXI .xpt files (required).
                     Parquet copies of the .xpt files are written next to them
                     by convert_xpt_to_parquet.py.
  T1DEXI_OUTPUT_DIR  root folder for all pipeline outputs
                     (default: <repo>/output).

Run `python config.py` to print the resolved paths and check which inputs exist.
See README.md for the order in which to run the pipeline steps.
"""

import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# USER CONFIGURATION
# ============================================================================

RAW_T1DEXI_PATH = os.environ.get('T1DEXI_RAW_DIR', '')
OUTPUT_ROOT = os.path.abspath(
    os.environ.get('T1DEXI_OUTPUT_DIR', os.path.join(CURRENT_DIR, 'output')))

# Name of the 8-hour non-exercise instance folder written by
# instance_feature_extraction.py and read by curate_data.py.
INSTANCE_FOLDER = 'non_exercise_8h'

# ============================================================================
# DERIVED PATHS
# ============================================================================

RAW_DATA_PATH = RAW_T1DEXI_PATH

# Written by get_netiob_cleaned_data.py
NETIOB_DATA_PATH = os.path.join(OUTPUT_ROOT, 'netiob_cleaned_data')

# Written by process_insulin.py and build_carb_lookup.py
CLEANED_DATA_PATH = os.path.join(OUTPUT_ROOT, 'cleaned_data')

# Written by get_netiob_cleaned_data.py (exercise sessions)
DATASETS_PATH = os.path.join(OUTPUT_ROOT, 'datasets')

# Written by split_parquets_by_user.py
BY_USER_DIR = os.path.join(OUTPUT_ROOT, 'by_user')

# Written by instance_feature_extraction.py and combine_by_user.py
OUTPUT_DATA_PATH = os.path.join(OUTPUT_ROOT, 'processed_data')

# Written by curate_data.py
FILTERING_OUTPUT_PATH = os.path.join(OUTPUT_ROOT, 'data_filtering')

# ============================================================================
# FILES
# ============================================================================

GLUCOSE_PARQUET  = os.path.join(NETIOB_DATA_PATH, 'glucose_data.parquet')
MEALS_PARQUET    = os.path.join(NETIOB_DATA_PATH, 'meal_data.parquet')
BOLUS_PARQUET    = os.path.join(NETIOB_DATA_PATH, 'bolus_insulin_data.parquet')
BASAL_PARQUET    = os.path.join(CLEANED_DATA_PATH, 'basal.parquet')
USER_DEVICE_PARQUET = os.path.join(NETIOB_DATA_PATH, 'user_device_data.parquet')
CARB_LOOKUP_PARQUET = os.path.join(CLEANED_DATA_PATH, 'carb_consumed_lookup.parquet')

EXERCISE_CSV     = os.path.join(DATASETS_PATH, 'exercise_data', 'exercise.csv')


def num_workers(cap=16):
    """Worker processes to use: T1DEXI_NUM_CPUS if set, else the CPUs available
    to this process (respects Slurm/cgroup affinity), capped at `cap`."""
    if os.environ.get('T1DEXI_NUM_CPUS'):
        return int(os.environ['T1DEXI_NUM_CPUS'])
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    return max(1, min(n, cap))


def require_raw_path():
    """Exit with a clear message if T1DEXI_RAW_DIR is not set or missing."""
    if not RAW_T1DEXI_PATH or not os.path.isdir(RAW_T1DEXI_PATH):
        raise SystemExit(
            'T1DEXI_RAW_DIR is not set or does not exist. Set it to the folder '
            'containing the raw T1DEXI .xpt files, e.g.\n'
            '  export T1DEXI_RAW_DIR=/path/to/T1DEXI')


# ============================================================================
# VALIDATION
# ============================================================================

def validate_paths():
    """Check that the inputs needed for instance extraction exist."""
    required_files = [
        ('Glucose (Parquet)',    GLUCOSE_PARQUET),
        ('Meals (Parquet)',      MEALS_PARQUET),
        ('Bolus (Parquet)',      BOLUS_PARQUET),
        ('Basal (Parquet)',      BASAL_PARQUET),
        ('Consumed-carb lookup', CARB_LOOKUP_PARQUET),
        ('Exercise (CSV)',       EXERCISE_CSV),
    ]

    missing = []
    found = []

    for name, path in required_files:
        if os.path.exists(path):
            found.append(name)
        else:
            missing.append(f"  [missing] {name}: {path}")

    if found:
        print(f"Found {len(found)} required input files:")
        for name in found:
            print(f"  [ok] {name}")

    if missing:
        print("\nWARNING: some input files were not found:")
        for m in missing:
            print(m)
        print("\nRun the pipeline steps in order (see README.md).")
        return False

    return True


if __name__ == '__main__':
    print("=" * 70)
    print("DATA PATH CONFIGURATION")
    print("=" * 70)
    print(f"\n  T1DEXI_RAW_DIR:    {RAW_T1DEXI_PATH or '(not set)'}")
    print(f"  T1DEXI_OUTPUT_DIR: {OUTPUT_ROOT}")
    print("\nFile validation:")
    validate_paths()
    print("=" * 70)
