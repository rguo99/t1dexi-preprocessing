"""
This file contains code for processing insulin data from AID 
(Automated Insulin Delivery) pumps in T1DEXI data. The primary 
purpose of this code is to chunk the data into 5-minute intervals 
for each AID pump type. This processing step is essential for 
preparing the data for further analysis, ensuring it is at a 
5-minute precision.

==================== HOW TO USE ==============================
Used as a module by process_insulin.py (chunk_procedure,
remove_oddities).
===============================================================

© Copyrights: CeADAR Connect Group @ CeADAR - Ireland's Centre for AI
Adapted from netIOB (Lewis et al., preprint); Apache License 2.0, see NOTICE.

Modifications in this copy: imported as a module only (the standalone CLI
and XPT export were removed, as were unused imports); rounding applied to the input frame; bolus chunking
falls back to FASTRESN when INSEXBOL is missing; overlap detection runs in
parallel per user.
"""

# Import libraries
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
pd.set_option('display.float_format', lambda x: '%.5f' % x)




def convert_to_iso8601(duration):
    """
    Converts processed duration in FADUR back
    to iso8601 format.
    
    Args:
    - duration: (Timestamp) Duration timestamp to be converted in format: e.g., '0 days HH:MM:SS
    
    Returns:
    - iso8601 Timestamp Format, e.g., P0Y0M0DT4H5M2S
    """
    
    if pd.notna(duration):
        duration_str = str(duration).replace('0 days ', '')    
        
        # Split the duration string into hours, minutes, and seconds
        hours, minutes, seconds = map(int, duration_str.split(':'))

        # Format the duration in ISO 8601 format
        iso8601_duration = f'P0Y0M0DT{hours}H{minutes}M{seconds}S'
    else:
        iso8601_duration = pd.NaT

    return iso8601_duration

def calculate_time_diff(group):
    """
    Calculate the time difference between consecutive timestamps for each group.

    Args:
    - group: (DataFrame) Grouped DataFrame where each group represents a subject's data.

    Returns:
    - (DataFrame) DataFrame with an additional column 'FADUR_new' containing the time difference between consecutive timestamps.
    """
    # Calculate the time difference between consecutive timestamps and pass values to a newly created colum 'FADUR_new'
    group['FADUR_new'] = group['FADTC'].diff().shift(-1).fillna(pd.Timedelta(seconds=0))
    return group

def finally_(dataframe):
    """
    Finalizes the chunking processing by converting timestamps,
    rounding of necessary column values, and converting the 
    duration column to iso8601 format. E.g., P0Y0M0DT4H5M2S
    
    Args:
    - dataframe (DataFrame) DataFrame to be processed
    
    Returns:
    - (DataFrame) Processed DataFrame
    """
    # removed the conversion of FADTC to nanoseconds because the datestamps are supposed to be ~2020 because reference is 1970, 
    # otherwise, the convert timestamp reference in base_utils is 1960, so FADTCs will be ~2010

    # # Convert FADTC column datetime values to integer representation (nanoseconds). 
    # # and divide the result by 10^9 to convert nanoseconds to seconds
    # dataframe['FADTC'] = (pd.to_datetime(dataframe['FADTC']).astype('int64') / 10**9).astype('int64')
    # Round of the values in FASTRESN, INSNMBOL, and INSEBOL to 2 decimal places
    dataframe[['FASTRESN', 'INSNMBOL', 'INSEXBOL']] = (dataframe[['FASTRESN', 'INSNMBOL', 'INSEXBOL']]).round(2)
    # COnvert FADUR column values to iso8601 format: P0Y0M0DT4H5M2S
    dataframe['FADUR'] = (dataframe['FADUR']).apply(convert_to_iso8601)
    
    return dataframe

def chunk_procedure(main_df, ins_type=None, device_type=None):
    """
    Chunk procedure to split insulin records into smaller chunks.

    Args:
    - main_df: (DataFrame) Main DataFrame containing insulin records.
    - ins_type: (str) Type of insulin ('BASAL' or 'BOLUS').
    - device_type: (str) Type of device ('TANDEM', '670G', or '770G').

    Returns:
    - (DataFrame) DataFrame with insulin records chunked and processed.
    """

    # Filtering records based on device type and insulin type
    if device_type in ['TANDEM', '670G'] and ins_type == 'BASAL':
        # Filtering main Frame for BASAL insulin type records for TANDEM and 670G devices
        filtered_df = main_df[(main_df['FADUR'] > pd.Timedelta(minutes=6)) & 
                               (main_df['FATEST'] == 'BASAL INSULIN') & 
                               (main_df['INSSTYPE'].isin(['scheduled', 'suspend', 'temp', 'automated']))]
    elif device_type in ['TANDEM', '670G'] and ins_type == 'BOLUS':
        # Filtering main Frame for BOLUS insulin type records for TANDEM and 670G devices
        filtered_df = main_df[(main_df['FADUR'] > pd.Timedelta(minutes=6)) &
                              (main_df['FACAT'] == 'BOLUS') &
                              (main_df['INSSTYPE'].isin(['dual/square', 'square'])) &
                              (main_df['INSEXBOL'] != 0)]
    elif device_type == '770G' and ins_type == 'BASAL':
        # Filtering main Frame for BASAL insulin type records for 770G device
        filtered_df = main_df[(main_df['FADUR'] > pd.Timedelta(minutes=6)) & 
                               (main_df['FATEST'] == 'BASAL INSULIN')]
                     
    
    # Creating an empty list to store the new rows after chunking
    new_rows = []
    # Iterating over each row in the filtered DataFrame
    for index, row in filtered_df.iterrows():
        # Calculate total seconds for the duration of insulin delivery
        total_seconds = row['FADUR'].total_seconds()
        # Calculate the number of 5-minutes chunks
        num_chunks = int(total_seconds / (5 * 60)) 
        # Calculate the remaining seconds after chunking
        remaining_seconds = total_seconds % (5 * 60)
        first_chunk_processed = False
        # Calculate insulin delivery rate per second if value not NaN
        if pd.notna(row['FASTRESN']):
            fastresn_per_second = row['FASTRESN'] / total_seconds
        insexbol_per_second = 0
        if pd.notna(row['INSEXBOL']):
            insexbol_per_second = row['INSEXBOL'] / total_seconds
        elif pd.notna(row['FASTRESN']):
            insexbol_per_second = row['FASTRESN'] / total_seconds

        # Chunking insulin records
        for i in range(num_chunks):
            new_row = row.copy()
            # Setting chunk duration to 5 minutes
            new_row['FADUR'] = pd.Timedelta(minutes=5)  
            # Adjusting timestamp for the chunked record
            new_row['FADTC'] = pd.Timestamp(row['FADTC']) + pd.Timedelta(minutes=i * 5) 
            # Updating the flag to indicate the first chunk has been processed
            first_chunk_processed = True if not first_chunk_processed else False            
            
            if device_type in ['TANDEM', '670G'] and ins_type == 'BASAL':
                # If it's the first chunk, set INSNMBOL to the original value, otherwise set it to 0
                new_row['INSNMBOL'] = row['INSNMBOL'] if not first_chunk_processed else 0
                # If it's the first chunk, set FADUR_original to the original value, otherwise set it to an empty string
                new_row['FADUR_original'] = row['FADUR'] if not first_chunk_processed else '' 
                # Adding '_chunk' suffix to INSSTYPE to indicate it's a chunked record
                new_row['INSSTYPE'] += '_chunk'
                # Calculating the amount of basal insulin for the chunk
                new_row['FASTRESN'] = fastresn_per_second * 5 * 60  

            elif device_type in ['TANDEM', '670G'] and ins_type == 'BOLUS':
                # If it's the first chunk, set INSNMBOL to the original value, otherwise set it to 0
                new_row['INSNMBOL'] = row['INSNMBOL'] if not first_chunk_processed else 0
                # If it's the first chunk, set INSEXBOL_original to the original value, otherwise set it to an empty string
                new_row['INSEXBOL_original'] = row['INSEXBOL'] if not first_chunk_processed else ''
                # If it's the first chunk, set FADUR_original to the original value, otherwise set it to an empty string
                new_row['FADUR_original'] = row['FADUR'] if not first_chunk_processed else ''
                # Adding '_chunk' suffix to INSSTYPE to indicate it's a chunked record
                new_row['INSSTYPE'] += '_chunk'
                # Setting FASTRESN to an empty string as it's not relevant for BOLUS insulin
                new_row['FASTRESN'] = '' 
                # Calculating the amount of bolus insulin for the chunk
                new_row['INSEXBOL'] = insexbol_per_second * 5 * 60  
            
            elif device_type == '770G' and ins_type == 'BASAL':
                # If it's the first chunk, set FASTRESN_original to the original value, otherwise set it to an empty string
                new_row['FASTRESN_original'] = row['FASTRESN'] if not first_chunk_processed else ''
                # If it's the first chunk, set FADUR_original to the original value, otherwise set it to an empty string
                new_row['FADUR_original'] = row['FADUR'] if not first_chunk_processed else ''
                # Adding 'basal_chunk' suffix to INSSTYPE to indicate it's a chunked basal record for 770G device 
                new_row['INSSTYPE'] = 'basal_chunk'
                # Calculating the amount of basal insulin for the chunk
                new_row['FASTRESN'] = fastresn_per_second * 5 * 60 
            
            # Appending the chunked row to the list of new rows
            new_rows.append(new_row)

        # Handling remaining seconds
        if remaining_seconds > 0:
            # Creating a copy of the current row for the last chunk
            new_row = row.copy()
            # Adjusting duration for the last chunk to reflect the remaining seconds
            new_row['FADUR'] = pd.Timedelta(seconds=remaining_seconds) 
            # Adjusting timestamp for the last chunk by adding remaining seconds to the last chunk timestamp
            new_row['FADTC'] = pd.Timestamp(row['FADTC']) + pd.Timedelta(seconds=num_chunks * 5 * 60)  
            
            if device_type in ['TANDEM', '670G'] and ins_type == 'BASAL':
                # Calculating the amount of basal insulin for the remaining time and assigning it to FASTRESN
                new_row['FASTRESN'] = fastresn_per_second * remaining_seconds 
                # Adding '_chunk' suffix to INSSTYPE to indicate it's a chunked record
                new_row['INSSTYPE'] += '_chunk' 

            elif device_type in ['TANDEM', '670G'] and ins_type == 'BOLUS':
                # Calculating the amount of bolus insulin for the remaining time and assigning it to INSEXBOL
                new_row['INSEXBOL'] = insexbol_per_second * remaining_seconds
                # Since it's the last chunk, setting INSNMBOL to 0 to indicate no additional basal insulin
                new_row['INSNMBOL'] = 0
                # Setting FASTRESN to an empty string as it's not relevant for BOLUS insulin
                new_row['FASTRESN'] = '' 
                # Adding '_chunk' suffix to INSSTYPE to indicate it's a chunked record
                new_row['INSSTYPE'] += '_chunk' 
            
            elif device_type == '770G' and ins_type == 'BASAL':
                # 'basal_chunk' as INSSTYPE to indicate it's a chunked basal record for 770G device
                new_row['INSSTYPE'] = 'basal_chunk'
                new_row['FASTRESN'] = fastresn_per_second * remaining_seconds 

            
            # Appending the last chunked row to the list of new rows
            new_rows.append(new_row)
            
    # Concatenating new rows and original DataFrame
    instype_chunk = pd.concat([main_df, pd.DataFrame(new_rows)], ignore_index=True)
    instype_chunk['FADUR'] = pd.to_timedelta(instype_chunk['FADUR'])

    # Removing unnecessary records based on device type and insulin type
    if device_type in ['TANDEM', '670G'] and ins_type == 'BASAL':
        rows_to_remove = instype_chunk[(instype_chunk['INSSTYPE'].isin(['scheduled', 'suspend', 'temp', 'automated'])) &
                                       (instype_chunk['FADUR'] > pd.Timedelta(minutes=5.5))]
    elif device_type in ['TANDEM', '670G'] and ins_type == 'BOLUS':
        rows_to_remove = instype_chunk[(instype_chunk['INSSTYPE'].isin(['dual/square', 'square'])) &
                                       (instype_chunk['INSEXBOL'] != 0) &
                                       (instype_chunk['FADUR'] > pd.Timedelta(minutes=5.5))]
    elif device_type == '770G' and ins_type == 'BASAL':
        rows_to_remove = instype_chunk[((instype_chunk['INSSTYPE'].isin(['normal'])) | 
                                       (instype_chunk['INSSTYPE'].isin(['']))) &
                                       (instype_chunk['FADUR'] > pd.Timedelta(minutes=5.5)) & 
                                       (instype_chunk['FACAT'] == 'BASAL')]
    
    instype_chunk = instype_chunk.drop(rows_to_remove.index)
    
    return instype_chunk.sort_values(by=['USUBJID', 'FADTC'])

def check_usubjid_for_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    df['FADTC'] = pd.to_datetime(df['FADTC'], errors='coerce')
    df['End_Time'] = df['FADTC'] + df['FADUR']
    user_data = (df[df['FATEST'] == 'BASAL INSULIN'].sort_values('FADTC').reset_index(drop=True).dropna(subset=['FADTC']))

    # Find overlapping timestamps
    overlaps = []
    for _, row in user_data.iterrows():
        overlap_condition = ((user_data['FADTC'] > row['FADTC']) & (user_data['FADTC'] < row['End_Time']) & 
                             (user_data['FASTRESN'].notna()) & (~user_data['FASTRESN'].eq('')) & 
                             (user_data['FACAT'] == 'BASAL'))
        overlap_timestamps = user_data.loc[overlap_condition, 'FADTC'].tolist()
        
        # Check if overlap_timestamps is truthy
        if overlap_timestamps:
            overlaps.append({'Start_FADTC': row['FADTC'], 'Overlapping_Timestamps': overlap_timestamps})
    
    return pd.DataFrame(overlaps)


def check_user_overlaps(args):
    """Worker function for parallel overlap detection."""
    userid, user_df = args
    overlap_df = check_usubjid_for_overlaps(user_df)
    if not overlap_df.empty:
        overlap_df['USUBJID'] = userid
        return overlap_df
    return None

def remove_oddities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unwanted USUBJIDs and Dates, as well as readjust wrongfully placed bolus types.
    Parallelized overlap detection for speed.
    """
    usubjid_to_remove = ['1039', '1046', '1439', '317', '422', '956']
    items_to_be_tossed = {'non_aid': {
                                '1016': ['2020-05-20'],
                                '946': ['2019-11-29', '2019-11-28']
                          },
                          'aid': {
                                '1076': ['2019-12-17'],
                                '1127': ['2019-07-23'],
                                '1193': ['2020-04-13', '2020-04-14'],
                                '1248': ['2020-02-11'],
                                '1542': ['2020-08-02'],
                                '1611': ['2019-07-30'],
                                '1615': ['2019-08-16'],
                                '183': ['2020-03-10'],
                                '550': ['2021-02-15', '2021-02-16', '2021-02-17', '2021-02-18', '2021-02-19',
                                        '2021-02-20', '2021-02-21', '2021-02-22'],
                                '576': ['2019-02-20', '2019-02-21'],
                                '692': ['2020-08-15'],
                                '945': ['2019-08-24']
                          }
                        }

    # Remove unwanted USUBJIDs
    df = df[~df['USUBJID'].isin(usubjid_to_remove)]

    # Remove bad dates
    for category in items_to_be_tossed.values():
        for usubjid, dates in category.items():
            dates_to_remove = pd.to_datetime(dates)
            df = df[~((df['USUBJID'] == usubjid) & ((df['FADTC'].dt.date).isin(dates_to_remove)))]

    truncated_df = df.copy()
    unique_users = truncated_df['USUBJID'].unique()

    print(f'  Checking overlaps for {len(unique_users)} users...')

    # Prepare args for parallel processing
    user_args = [
        (userid, truncated_df[truncated_df['USUBJID'] == userid])
        for userid in unique_users
    ]

    # Parallel overlap detection with progress bar
    n_cpus = min(cpu_count(), 8)
    all_overlaps = []
    with Pool(processes=n_cpus) as pool:
        for result in tqdm(
            pool.imap(check_user_overlaps, user_args),
            total=len(user_args),
            desc='  Detecting overlaps'
        ):
            if result is not None:
                all_overlaps.append(result)

    if all_overlaps:
        all_overlaps_df = pd.concat(all_overlaps, ignore_index=True)
        all_overlaps_df = all_overlaps_df.explode('Overlapping_Timestamps').reset_index(drop=True)
        all_overlaps_df['Overlapping_Timestamps'] = pd.to_datetime(all_overlaps_df['Overlapping_Timestamps'])
        all_overlaps_df = all_overlaps_df.rename(columns={'Overlapping_Timestamps': 'FADTC'})
        all_overlaps_df['FADTC'] = pd.to_datetime(all_overlaps_df['FADTC'])
        truncated_df['FADTC'] = pd.to_datetime(truncated_df['FADTC'])

        df['USUBJID'] = df['USUBJID'].astype(str)
        all_overlaps_df['USUBJID'] = all_overlaps_df['USUBJID'].astype(str)
            
        overlap_keys = set(zip(all_overlaps_df['USUBJID'], all_overlaps_df['FADTC']))
        mask = pd.Series([
            (str(u), t) not in overlap_keys 
            for u, t in zip(truncated_df['USUBJID'], truncated_df['FADTC'])
        ], index=truncated_df.index)
        truncated_df = truncated_df[mask | (truncated_df['FATEST'] != 'BASAL INSULIN')]

        all_adjusted_df = pd.DataFrame(truncated_df)

    else:
        all_adjusted_df = pd.DataFrame(truncated_df)

    # Adjust BOLUS INSULIN values
    condition = ((truncated_df['FATEST'] == 'BOLUS INSULIN') &
                 (truncated_df['INSNMBOL'].fillna(0) == 0) &
                 (truncated_df['INSEXBOL'].fillna(0) == 0) &
                 (truncated_df['FASTRESN'] > 0))

    all_adjusted_df.loc[condition, 'INSNMBOL'] = all_adjusted_df['FASTRESN']
    all_adjusted_df['USUBJID'] = all_adjusted_df['USUBJID'].astype(int).astype(str)

    return all_adjusted_df
