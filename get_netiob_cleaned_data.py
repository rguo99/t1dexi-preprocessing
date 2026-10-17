"""
get_netiob_cleaned_data.py
==========================
Step 2 of the pipeline. Reads raw T1DEXI data via T1DEXI_Utils and writes
cleaned parquet files to $T1DEXI_OUTPUT_DIR/netiob_cleaned_data/, plus the
exercise-session table used to exclude exercise windows.

Prerequisite: run convert_xpt_to_parquet.py first for faster loading.

Usage:
    python get_netiob_cleaned_data.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RAW_DATA_PATH, NETIOB_DATA_PATH, EXERCISE_CSV, require_raw_path
from netiob.base_utils import T1DEXI_Utils

utils = None


def process_user_device_data():
    """Device table (USUBJID, DXTRT, DEVICE, device_type) for every participant.

    device_type is 'AID' for automated-insulin-delivery systems and 'non-AID'
    otherwise (manual pumps, predictive-suspend pumps and MDI); DEVICE keeps
    the finer category, e.g. 'non-AID (MDI)'.
    """
    user_device_data = utils.get_device_data()[0]
    user_device_data['device_type'] = pd.NA
    user_device_data.loc[user_device_data.DEVICE == 'AID', 'device_type'] = 'AID'
    user_device_data.loc[user_device_data.DEVICE != 'AID', 'device_type'] = 'non-AID'

    print('  Participants by device category:')
    for category, n in user_device_data['DEVICE'].value_counts().items():
        print(f'    {category}: {n}')

    user_device_data.to_parquet(os.path.join(NETIOB_DATA_PATH, 'user_device_data.parquet'), index=False)
    return user_device_data


def add_device_category(df, user_device_data):
    """Left-join the device columns onto a per-participant table."""
    df = pd.merge(df, user_device_data, how='left', on='USUBJID')
    return df



def generate_exercise_csv(exercise_data):
    """
    Generate exercise CSV for instance extraction.
    Converts raw exercise parquet to a clean CSV with start/end times.
    """
    exercise_processed = exercise_data[['USUBJID', 'PRSTDTC', 'PLNEXDUR']].copy()
    exercise_processed.rename(columns={
        'USUBJID': 'id',
        'PRSTDTC': 'ex_start_time',
        'PLNEXDUR': 'ex_duration'
    }, inplace=True)
    exercise_processed['ex_start_time'] = pd.to_datetime(exercise_processed['ex_start_time'])
    exercise_processed['ex_duration'] = exercise_processed['ex_duration'].astype(int)
    exercise_processed['ex_end_time'] = (
        exercise_processed['ex_start_time'] +
        pd.to_timedelta(exercise_processed['ex_duration'], unit='m')
    )
    exercise_processed = exercise_processed.reset_index(drop=True)
    exercise_processed.index.name = 'ex'
    exercise_processed = exercise_processed.reset_index()

    os.makedirs(os.path.dirname(EXERCISE_CSV), exist_ok=True)
    exercise_processed.to_csv(EXERCISE_CSV, index=False)
    print(f'  Exercise CSV saved: {exercise_processed.shape[0]} exercises -> {EXERCISE_CSV}')
    return exercise_processed

def main():
    """Write the cleaned glucose, meal, insulin, exercise and device tables."""
    global utils
    require_raw_path()
    utils = T1DEXI_Utils(RAW_DATA_PATH + '/')
    os.makedirs(NETIOB_DATA_PATH, exist_ok=True)

    print('=' * 60)
    print('GETTING NETIOB CLEANED DATA')
    print('=' * 60)

    print('\nProcessing device data...')
    user_device_data = process_user_device_data()

    print('Processing glucose data...')
    glucose_data = utils.get_glucose_data()
    glucose_data = add_device_category(glucose_data, user_device_data)
    glucose_data.to_parquet(os.path.join(NETIOB_DATA_PATH, 'glucose_data.parquet'), index=False)

    print('Processing meal data...')
    meal_data = utils.get_meal_data()
    meal_data = add_device_category(meal_data, user_device_data)
    meal_data.to_parquet(os.path.join(NETIOB_DATA_PATH, 'meal_data.parquet'), index=False)

    print('Processing insulin data...')
    insulin_data, basal_insulin_data, bolus_insulin_data = utils.get_insulin_data()
    insulin_data = add_device_category(insulin_data, user_device_data)
    basal_insulin_data = add_device_category(basal_insulin_data, user_device_data)
    bolus_insulin_data = add_device_category(bolus_insulin_data, user_device_data)
    basal_insulin_data.to_parquet(os.path.join(NETIOB_DATA_PATH, 'basal_insulin_data.parquet'), index=False)
    bolus_insulin_data.to_parquet(os.path.join(NETIOB_DATA_PATH, 'bolus_insulin_data.parquet'), index=False)

    print('Processing exercise data...')
    exercise_data = utils.get_exercise_data()
    exercise_data = add_device_category(exercise_data, user_device_data)
    exercise_data.to_parquet(os.path.join(NETIOB_DATA_PATH, 'exercise_data.parquet'), index=False)

    print('Generating exercise CSV...')
    generate_exercise_csv(exercise_data)

    print('\nAll data written to:', NETIOB_DATA_PATH)
    print('=' * 60)


if __name__ == '__main__':
    main()
