"""
This file is used to pre-process and load T1DEXI data for 
further analysis. Some key data functions have been explicitly 
designed to pull data directly from T1DEXI, including:

    - Exercise Data: get_exercise_data()
    - Glucose Data: get_glucose_data()
    - Insulin Data: get_insulin_data_raw()
    - Meal Data: get_meal_data()    
    - Device Data: get_device_data()

==================== HOW TO RUN ==============================

- STEP 1: from base_utils import T1DEXI_Utils
- STEP 2: Call pre-defined functions to pull data from T1DEXI 
either by:

    (i) using the instance of T1DEXI_Utils, e.g., 
        utils = T1DEXI_Utils()
        meal_dataFrame = utils.get_meal_data()
    (ii) direct use, e.g.,
        T1DEXI_Utils().get_meal_data()

To get any other data file apart from the functions already 
provided, use any of the folowing (for example):

    utils._get_data('DM')[0] 
    T1DEXI_Utils()._get_data('DM')[0]

*DM* is the data file name. It can be any other filename. 
(Note: the file name is used without the .xpt)
================================================================

© Copyrights: CeADAR Connect Group @ CeADAR - Ireland's Centre for AI
Adapted from netIOB (Lewis et al., preprint); Apache License 2.0, see NOTICE.

Modifications in this copy: _get_data() reads a parquet copy of each domain
when one exists; get_meal_data() keeps only FAMLPM rows with
FACAT == 'CONSUMED'; unused methods (demographics, chunked-insulin reader,
XPT export) and imports were removed.
"""

import pandas as pd
from datetime import datetime, timedelta
import os
import re
import pyreadstat
import itertools


class T1DEXI_Utils:
    def __init__(self, DATA_PATH='T1DEXI_dataset/'):    # Define data directory path in param, e.g., DATA_PATH='PATH_TO_T1DEXI_FILES'
        """
        Defines major variables key to the processing of the T1DEXI dataset
        
        Args:
        - None

        Returns:
        - None
        """       
        self.DATA_PATH = DATA_PATH   
        
        # Define list variable for all timestamp columns
        self.data_datetime_list = ['RFICDTC', 'BRTHDTC', 'DXDTC', 'LBDTC', 'MLDTC', 'NVDTC', 'NVENDTC', 'PRSTDTC', 'QSDTC', 
                                   'PRDTC', 'SCDTC', 'RPDTC', 'DMDTC', 'CMSTTPT', 'CMENTPT', 'VSDTC', 'VSENDTC', 'FADTC', 'DATETIME']
        # Define list variable for all AID Insulin Pumps
        self.aid_device_list = ['TANDEM T:SLIM X2 WITH CONTROL IQ', 'MEDTRONIC 670G IN AUTO MODE', 'MEDTRONIC 770G IN AUTO MODE']
        # Define list variables for all non-AID insulin pumps
        self.non_aid_device_list = ['INSULET OMNIPOD INSULIN MANAGEMENT SYSTEM', 'MEDTRONIC 670G', 'INSULET OMNIPOD DASH', 'MEDTRONIC 551 (530G)', 
                                    'MEDTRONIC 630G', 'MEDTRONIC 670G IN MANUAL MODE', 'MEDTRONIC 751 (530G)', 'MEDTRONIC PARADIGM 522', 
                                    'MEDTRONIC PARADIGM 523 (REVEL)', 'MEDTRONIC PARADIGM 723 (REVEL)', 'TANDEM T:SLIM', 'TANDEM T:SLIM X2', 
                                    'MEDTRONIC 770G', 'MEDTRONIC 511', 'MEDTRONIC PARADIGM 512', 'MEDTRONIC PARADIGM 551', 'MEDTRONIC REVEL', 
                                    'TANDEM T:FLEX', 'INSULIN PUMP']
        # Define list variable for MDI pumps
        self.mdi_device_list = ['MULTIPLE DAILY INJECTIONS']
        # Define list variable for all non_AID predictive suspend pumps
        self.non_aid_predictive_list = ['MEDTRONIC 640G', 'TANDEM T:SLIM X2 WITH BASAL IQ']

    def convert_timestamp(self, timestamps_col):
        """
        Converts timestamp columns in the input DataFrame from seconds since 1960-01-01
        to the format 'YYYY-MM-DD HH:MM:SS'

        Args:
        timestamps_col : timestamp column(s) to be converted
        
        Returns:
        converted_timestamp: (Datetime) timestamp column(s) converted to 'YYYY-MM-DD HH:MM:SS' format.
        """
        # Check the type of timestamps_col (If it's a Series and not of type pd.Timestamp)
        if isinstance(timestamps_col, pd.Series) and timestamps_col.dtype != pd.Timestamp:
            # Error handling
            try:
                # Convert timestamp to datetime format if not null or NaT
                timestamps_col = timestamps_col.apply(lambda x: datetime(1960, 1, 1) + timedelta(seconds=x) if pd.notnull(x) else pd.NaT)
            except TypeError:
                timestamps_col = pd.to_datetime(timestamps_col)
        
        # Otherwise if timestamps_col is in float format
        elif isinstance(timestamps_col, float):
            timestamps_col = datetime(1960, 1, 1) + timedelta(seconds=timestamps_col)
    
        return timestamps_col

    def iso8601_duration_to_timedelta(self, duration_str):
        """
        Converts duration in ISO8601 (P0Y0M0DT4H0M0S) format to HH:MM:SS

        Args:
        - duration_str: (str) Duration String Value

        Returns:
        - Time format (HH:MM:SS)
        """
        # Regular expression to extract components
        regex = r'P(?:([0-9]+)Y)?(?:([0-9]+)M)?(?:([0-9]+)D)?(?:T(?:([0-9]+)H)?(?:([0-9]+)M)?(?:([0-9]+)S)?)?'      
        matches = re.match(regex, duration_str)         # Match strings in the input duration string
        
        if matches:     # If there's a match            
            # Extract time related components 
            years = int(matches.group(1)) if matches.group(1) else 0
            months = int(matches.group(2)) if matches.group(2) else 0
            days = int(matches.group(3)) if matches.group(3) else 0
            hours = int(matches.group(4)) if matches.group(4) else 0
            minutes = int(matches.group(5)) if matches.group(5) else 0
            seconds = int(matches.group(6)) if matches.group(6) else 0

            return timedelta(hours=hours, minutes=minutes, seconds=seconds)     # Convert to timedelta
        else:       # If no match, return 00:00:00
            return timedelta(hours=0, minutes=0, seconds=0)

    def clean_dataframe(self, data):
        """
        Cleans the input DataFrame by decoding, stripping unwanted characters, and converting the timestamps.

        Args:
        - data: (DataFrame) Input DataFrame to be cleaned.

        Returns:
        - cleaned_data: (DataFrame) Cleaned DataFrame with unwanted characters removed from object-type columns,
                       rows containing unwanted characters removed, and timestamps converted.
        """
        unwanted_xters_list = ['%']     # Unwanted character list

        # Strip the object-type characters "b'" from the data
        cleaned_data = data.map(lambda x: x.decode('latin-1') if isinstance(x, bytes) else x)
        # Remove the unwanted characters
        cleaned_data = cleaned_data[~cleaned_data.isin(unwanted_xters_list).any(axis=1, skipna=True)]

        for cols in cleaned_data.columns:     # Iterate through the dataframe columns
            # Check if processing column is in the list of timestamp columns
            if cols in self.data_datetime_list:
                # If yes, call the implement the convert_timestamp method on it.
                cleaned_data[cols] = cleaned_data[cols].map(lambda x: self.convert_timestamp(x))
        
        # Return clean dataframe
        return cleaned_data

    def _get_data(self, data_label, read_csv=None):
        """
        Reads the data from a SAS file, calls the clean_dataframe() function to strip, clean, 
        dropna, and convert timestamps, and returns the cleaned DataFrame along with the count 
        of number of data subjects in the DatFrame.
        
        Note: We provided options to use XPORT or PYREADSTAT libraries. Our implementation is 
        etup to use pyreadstat

        Args:
        - DATA_PATH: (FILE_DIR) File Directory Path to T1DEXI Dataset
        - data_label: (str) Filename without .xpt. Example: 'FACM', 'ML', etc.
            
        Returns:
        - data_label_df: (DataFrame) DataFrame after cleaning, converting timestamp columns, and removing NaN rows.
        - data_subject_count: (int) Count of unique data subjects in the prepared DataFrame.
        """

        # Check for parquet first (fast)
        parquet_path = f'{self.DATA_PATH}{data_label}.parquet'
        if os.path.exists(parquet_path):
            data_label_df = pd.read_parquet(parquet_path)
            if 'USUBJID' in data_label_df.columns:
                data_subject_count = data_label_df['USUBJID'].nunique()
            else:
                data_subject_count = None
            return data_label_df, data_subject_count
        
        # Fall back to xpt (slow, first time only)
        if read_csv:
            data_label_df = pd.read_csv(f'{self.DATA_PATH}{data_label}.csv')
        else:
            data_label_df, metadata = pyreadstat.read_xport(f'{self.DATA_PATH}{data_label}.xpt')
        
        data_label_df = self.clean_dataframe(data_label_df)
        if 'USUBJID' in data_label_df.columns:
            data_subject_count = data_label_df['USUBJID'].nunique()
        else:
            data_subject_count = None
        return data_label_df, data_subject_count

    def categorize_device_type(self, device_name):
        """
        Categorizes device types in the Device (DX) as AID and non-AID pumps

        Args:
        - device_name: (str) Takes in name of device

        Returns:
        - (str) Category of device type
        """
        # Check the name (type) of device and return the category
        if device_name in self.aid_device_list:
            return 'AID'
        elif device_name in self.non_aid_device_list:
            return 'non-AID'
        elif device_name in self.mdi_device_list:
            return 'non-AID (MDI)'
        elif device_name in self.non_aid_predictive_list:
            return 'non-AID (Predictive Suspend)'    

    def get_device_data(self, device_category=None):
        """
        Retrieves device data from the prepared DataFrame, categorizes the device type,
        and filters out certain device types, and returns (1) DataFrames with user id, 
        device type and device category, (2) DataFrame filtered by pump users category,
        and (3) list of all device types.

        Args:
        - device_category: (str) Device type, e.g., 'aid', 'non_aid'
                           Note: Default = None (i.e., all categories will be returned)
        Returns:
        - device_category_data: (DataFrame) DataFrame filtered by only the entered device category
        - all_device_list: (list) List of all device types
        """
        # Get the device type data and streamline to only two columns - 'USUBJID' and 'DXTRT'
        device_data = self._get_data('DX')[0][['USUBJID', 'DXTRT']]
        # Filter out device type 'INSLIN PUMP' and 'CLOSED LOOP INSULIN PUMP' as they are both generic device type names
        device_data = device_data[~device_data['DXTRT'].isin(['INSULIN PUMP', 'CLOSED LOOP INSULIN PUMP'])]

        # [Discretional]. We observed that subject ID: 766 device type should be 'MEDTRONIC 770G IN AUTO MODE' and we updated as such
        device_data.loc[device_data['USUBJID'] == '766', 'DXTRT'] = 'MEDTRONIC 770G IN AUTO MODE'
        # Create a new column and populate with the appropriate device category using the categorize_device_type() function
        device_data['DEVICE'] = device_data['DXTRT'].apply(self.categorize_device_type)

        # Create a fileter for the 'AID' and 'non-AID' device categories
        if device_category == 'aid':
            # Get only AID device data
            device_category_data = device_data[device_data['DXTRT'].isin(self.aid_device_list)]
        elif device_category == 'non_aid':
            # Get only non-AID device data
            # Note that all non-AID categories are combined here
            extend_non_aid_list = itertools.chain(self.non_aid_device_list, self.mdi_device_list, self.non_aid_predictive_list)
            device_category_data = device_data[device_data['DXTRT'].isin(extend_non_aid_list)]
        else:   # Get all device categories
            device_category_data = device_data
        
        # Get the list of all devices in the DataFrame
        all_device_list = device_data['DXTRT'].unique()

        # Return dataframe classified by devices categories, and list of all devices, respectively
        return device_category_data, all_device_list

    def get_glucose_data(self, device_category=None):
        """
        Retrieves Glucose data from the cleaned DataFrame and returns (1)
        a streamlined glucose dataframe, and (2) glucose data based on
        pump users categories.

        Args:
        - device_category: (str) Device type, e.g., 'aid', 'non_aid'
                           Note: Default = None (i.e., all categories will be returned)
        Returns:      
        - device_category_glucose_df: (DatFrame) Filtered glucose data based on input device category
        """ 
        # Get the glucose data and streamline to few important columns
        glucose_data = self._get_data('LB')[0][['USUBJID', 'LBORRES', 'LBORRESU', 'LBDTC']]
        # Get user ids of categories of device, e.g., 'aid', 'non_aid' (Depending on the input cat.)
        user_device_data = self.get_device_data(device_category=device_category)[0]['USUBJID']
        # Filter the glucose data based on users ids extracted for device categories    
        device_category_glucose_df = glucose_data[glucose_data['USUBJID'].isin(user_device_data)] 

        return device_category_glucose_df

    def get_meal_data(self, device_category=None):
        """
        Retrieves Meal data from the cleaned DataFrame and returns (1)
        a streamlined meal dataframe, and (2) meal data based on
        pump users categories.

        Args:
        - device_category: (str) Device type, e.g., 'aid', 'non_aid'
                           Note: Default = None (i.e., all categories will be returned)
        Returns:
        - device_category_meal_df: (DatFrame) Filtered meal data based on input device category
        """  

        print('Proccessing rescue carb data...')
        # Get the meal data and streamline to few important columns
        rescue_carb_data = self._get_data('ML')[0][['USUBJID', 'MLTRT', 'MLCAT', 'MLDOSE', 'MLDTC']]
        rescue_carb_data = rescue_carb_data.dropna(subset=['MLDOSE'])

        print('Processing meal data...')
        # FAMLPM records each meal's carbohydrate total as a TAKEN / RETURNED /
        # CONSUMED triple (T1DEXI photo plate-waste protocol; the status is in
        # FACAT, while FAOBJ holds the meal-time label, e.g. 'Dinner'). Only
        # CONSUMED reflects intake, so keep only those rows; otherwise carb
        # totals are inflated two- to three-fold.
        per_meal_data = self._get_data('FAMLPM', read_csv=True)[0][['USUBJID', 'FATEST', 'FAOBJ', 'FACAT', 'FASTRESN', 'FADTC']]
        per_meal_data = per_meal_data.dropna(subset=['FASTRESN'])
        per_meal_data = per_meal_data[per_meal_data.FACAT == 'CONSUMED']
        per_meal_data = per_meal_data[per_meal_data.FATEST ==  'Dietary Total Carbohydrate']
        per_meal_data = per_meal_data.drop(columns=['FACAT'])
        per_meal_data.rename(columns={'FAOBJ': 'MLTRT', 'FATEST': 'MLCAT', 'FASTRESN': 'MLDOSE', 'FADTC': 'MLDTC'}, inplace=True)
        per_meal_data.USUBJID = per_meal_data.USUBJID.astype('str')
        meal_data = pd.concat([per_meal_data, rescue_carb_data])

        # Get user ids of categories of device, e.g., 'aid', 'non_aid' (Depending on the input cat.)
        user_device_data = self.get_device_data(device_category=device_category)[0]['USUBJID']
        # Filter the meal data based on users ids extracted for device categories    
        device_category_meal_df = meal_data[meal_data['USUBJID'].isin(user_device_data)]

        # Return meal DataFrame filtered by the entered device categories
        return device_category_meal_df


    def get_exercise_data(self, device_category=None):
        """
        Retrieves Exercise data from the cleaned DataFrame and returns (1)
        a streamlined exercise dataframe, and (2) exercise data based on
        pump users categories.

        Args:
        - device_category: (str) Device type, e.g., 'aid', 'non_aid'
                           Note: Default = None (i.e., all categories will be returned)
        Returns:
        - device_category_exercise_df: (DatFrame) Filtered exercise data based on input device category
        """
        # Get the exercise data and streamline to few important columns  
        exercise_data = self._get_data('PR')[0][['USUBJID', 'PRTRT', 'PRSTDTC', 'PRTRTC', 'PLNEXDUR', 'EXCINTSY', 'HRSLSTML', 'RESQCARB']]
        # Get user ids of categories of pump users, e.g., 'aid', 'non_aid' (Depending on the input cat.)
        user_device_data = self.get_device_data(device_category=device_category)[0]['USUBJID']
        # Filter the exercise data based on users ids extracted for device categories    
        device_category_exercise_df = exercise_data[exercise_data['USUBJID'].isin(user_device_data)]        

        # Return Exercise DataFrame filtered by the entered device category
        return device_category_exercise_df

    def get_insulin_data(self, device_category=None):
        """
        Retrieves Insulin data from the cleaned DataFrame and returns (1)
        a streamlined insulin dataframe, and (2) insulin data based on
        pump users categories.

        Args:
        - device_category: (str) Device type, e.g., 'aid', 'non_aid'
                           Note: Default = None (i.e., all categories will be returned)
        Returns:
        - device_category_insulin_df: (DatFrame) Filtered insulin data based on input device category
        - basal_data: (DataFrame) Filtered dataframe with only basal data
        - bolus_data: (DataFrame) Filtered dataframe with only bolus data
        """
        # Get the insulin data and streamline to few important columns  
        insulin_data = self._get_data('FACM')[0][['USUBJID', 'FATEST', 'FACAT', 'FADTC', 'FASTRESN', 'FADUR', 'INSSTYPE', 'INSNMBOL', 'INSEXBOL']]

        # Convert the iso8601 duration format to a readable form and remove the '0 days' string prepended to the output
        insulin_data['FADUR'] = (insulin_data['FADUR'].apply(self.iso8601_duration_to_timedelta)).astype(str).str.replace('0 days ', '')

        # Round the unit values in the columns ['FASTRESN', 'INSNMBOL', 'INSEXBOL'] to 2 decimal places (takes care of scientific notation formats)  
        insulin_data[['FASTRESN', 'INSNMBOL', 'INSEXBOL']] = (insulin_data[['FASTRESN', 'INSNMBOL', 'INSEXBOL']]).round(2)
        insulin_data['FADTC'] = pd.to_datetime(insulin_data['FADTC'])

        basal_data = insulin_data[insulin_data['FACAT'] == 'BASAL']     # Extract only BASAL records
        bolus_data = insulin_data[insulin_data['FACAT'] == 'BOLUS']     # Extract only BOLUS records
        
        # Get user ids of categories of pump users, e.g., 'aid', 'non_aid' (Depending on the input cat.)
        user_device_data = self.get_device_data(device_category=device_category)[0]['USUBJID']
        
        # Filter the insulin data based on users ids 
        # extracted for pump user categories.    
        device_category_insulin_df = insulin_data[insulin_data['USUBJID'].isin(user_device_data)]

        # [Optional] If BASAL/BOLUS records based on user category is sought, uncomment the code line below,
        # change the 'BASAL' or 'BOLUS' values accordingly, and return the variable
        #device_category_insulin_type_df = device_category_insulin_df[device_category_insulin_df['FACAT'] == 'BASAL']

        # Return Exercise DataFrame filtered by entered device category, basal only records, bolus only records
        return device_category_insulin_df, basal_data, bolus_data
