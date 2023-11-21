import argparse
import pandas as pd
import os
import numpy as np
from dateutil import parser
import tqdm
from sklearn.impute import KNNImputer

# Dictionary mapping ID to country index
ID_Country = {
    # Format: 'ID': Index
    '10YES-REE------0': 0,  # Spain
    '10Y1001A1001A92E': 1,  # United Kingdom
    '10Y1001A1001A83F': 2,  # Germany
    '10Y1001A1001A65H': 3,  # Denmark
    '10YHU-MAVIR----U': 4,  # Hungary
    '10YSE-1--------K': 5,  # Sweden
    '10YIT-GRTN-----B': 6,  # Italy
    '10YPL-AREA-----S': 7,  # Poland
    '10YNL----------L': 8   # Netherlands
}

def process_energy_data(file_path):
    """
    Processes energy data from CSV files located at `file_path`.
    It reads, cleans, aggregates, and reformats the data into a consistent format.

    :param file_path: Path to the directory containing CSV files.
    :return: A pandas DataFrame containing the processed data.
    """
    # List of all file paths in the given directory
    files_to_load = [os.path.join(file_path, file) for file in os.listdir(file_path)]
    df = pd.DataFrame()

    # Iterating over each file
    for f in tqdm.tqdm(files_to_load):
        # Skip non-csv and test files
        if not f.endswith('csv'):
            continue

        # Read and process each CSV file
        df_tmp = pd.read_csv(f)
        df_tmp = df_tmp.drop('EndTime', axis=1)
        df_tmp.iloc[:, -1] = df_tmp.iloc[:, -1].interpolate(method='linear', limit_direction='both')
        df_tmp['StartTime'] = df_tmp['StartTime'].str.replace(r'\+00:00Z', '', regex=True)
        df_tmp['StartTime'] = pd.to_datetime(df_tmp['StartTime'], format='%Y-%m-%dT%H:%M')
        df_tmp.set_index('StartTime', inplace=True)
        agg_dict = {col: 'sum' if col in ['quantity', 'Load'] else 'first' for col in df_tmp.columns}
        df_tmp = df_tmp.resample('H').agg(agg_dict)
        df_tmp.reset_index(inplace=True)
        df = pd.concat([df, df_tmp], ignore_index=True)
    # Filtering specific PsrType values
    filter_conditions = [(df.PsrType != 'B0' + str(i)) for i in range(2, 9)] + \
                        [(df.PsrType != 'B' + str(i)) for i in range(14, 21)]

    combined_condition = filter_conditions[0]
    for condition in filter_conditions[1:]:
        combined_condition = combined_condition & condition

    df = df[combined_condition]

    # Creating a DataFrame for results
    df_ = pd.DataFrame(columns=['ID', 'StartTime', 'UnitName', 'Biomass', 'Geo', 'Hydro Pumped Storage',
                                'Hydro RP', 'Hydro WR', 'Marine', 'Other ',
                                'Solar', 'Wind Offshore', 'Wind Onshore', 'Load'])

    # Processing and aggregating the data
    result_data = []
    for _, group in df.groupby(['AreaID', 'StartTime']):
        # Extracting and mapping data
        country_id = ID_Country.get(group['AreaID'].iloc[0], 0)
        start_time = group['StartTime'].iloc[0]
        unit_name = group['UnitName'].iloc[0]
        quantities = dict(zip(group['PsrType'], group['quantity']))

        # Assigning quantities to respective energy types
        energy_types = ['B01', 'B09', 'B10', 'B11', 'B12', 'B13', 'B15', 'B16', 'B18', 'B19']
        energy_values = [quantities.get(etype, 0) for etype in energy_types]

        # Calculating load
        load = group['Load'].dropna().iloc[0] if ('Load' in group.columns) and (not group['Load'].dropna().empty) else 0

        # Appending the processed data for this group
        result_data.append([country_id, start_time, unit_name] + energy_values + [load])

    # Creating a DataFrame from the aggregated data
    df_ = pd.DataFrame(result_data, columns=['ID', 'StartTime', 'UnitName', 'Biomass', 'Geo',
                                             'Hydro Pumped Storage', 'Hydro RP',
                                             'Hydro WR', 'Marine', 'Other ', 'Solar', 
                                             'Wind Offshore', 'Wind Onshore', 'Load'])

    return df_

# Functions for data cleaning

def missing_values(df):
    """
    Removes rows where 'Load' is 0.
    
    :param df: DataFrame to clean.
    :return: DataFrame without rows where 'Load' is 0.
    """
    df_cleaned = df.loc[df['Load'] != 0]
    return df_cleaned

def duplicates(df):
    """
    Removes duplicate rows from the DataFrame.
    
    :param df: DataFrame to clean.
    :return: DataFrame without duplicates.
    """
    df_cleaned = df.drop_duplicates()
    return df_cleaned

def remove_outliers_iqr(df, column_name, threshold=1.5):
    """
    Removes outliers based on the Interquartile Range (qr) method.
    
    :param df: DataFrame to clean.
    :param column_name: Name of the column to check for outliers.
    :param threshold: Multiplier for qr to define bounds (default 1.5).
    :return: DataFrame with outliers removed.
    """
    Q1 = df[column_name].quantile(0.2)
    Q3 = df[column_name].quantile(0.8)
    qr = Q3 - Q1
    
    upper_bound = Q3 + threshold * qr
    lower_bound = Q1 - threshold * qr
    

    cleaned_df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    return cleaned_df

def clean_data(df):
    """
    Main function to clean the data.
    It currently only removes duplicates, but can be expanded to include other cleaning functions.
    
    :param df: DataFrame to clean.
    :return: Cleaned DataFrame.
    """
    df_cleaned = duplicates(df)
    return df_cleaned

def preprocess_data(df):
    """
    Preprocesses the data to calculate energy surplus for each country and timestamp.
    
    :param df: DataFrame to preprocess.
    :return: DataFrame with processed data.
    """
    df['StartTime'] = pd.to_datetime(df['StartTime'], format='%Y-%m-%dT%H:%M')
    aggregated_values = []

    column_name = {
        0: 'overload_SP',
        1: 'overload_UK',
        2: 'overload_DE',
        3: 'overload_DK',
        4: 'overload_HU',
        5: 'overload_SE',
        6: 'overload_IT',
        7: 'overload_PO',
        8: 'overload_NL',
    }   
    unique_timestamps = sorted(df['StartTime'].unique())
    for timestamp in unique_timestamps:
        timestamp_data = df[df['StartTime'] == timestamp]
        result = timestamp_data.iloc[:, 3:13].sum(axis=1) - timestamp_data['Load']
        total_sum = result.groupby(timestamp_data['ID']).sum().to_dict()
        total_sum_mapped = {column_name.get(key, key): value for key, value in total_sum.items()}
        aggregated_values.append({'StartTime': timestamp, **total_sum_mapped})

    df_processed = pd.DataFrame(aggregated_values).set_index('StartTime').fillna(0)
    df_processed = df_processed.reindex(sorted(df_processed.columns), axis=1)

    df_processed = df_processed.drop("overload_UK",axis=1)
    df_processed = fix(df_processed)

    return df_processed

def fix(df):
    """
    Further processes the DataFrame to predict surplus and clean data.
    
    :param df: DataFrame to process.
    :return: Processed DataFrame.
    """
    df_copy = df.copy()
    df['overload'] = df_copy.idxmax(axis=1)
    for column_name in df.columns[:-1]:
        df.loc[(df[column_name] >= -1) & (df[column_name] <= 0), column_name] = 0

    column_name = { #todo add it global
        'overload_SP' : 0,
        'overload_UK': 1,
        'overload_DE': 2,
        'overload_DK': 3,
        'overload_HU': 4,
        'overload_SE': 5,
        'overload_IT': 6,
        'overload_PO': 7,
        'overload_NL': 8,
    }   
    df['overload'] = df['overload'].map(column_name)
    df['overload'] = df['overload'].shift(-1)
    df = df.iloc[1:-1]

    return df

def save_data(df, output_file):
    """
    Saves the DataFrame to a CSV file.
    
    :param df: DataFrame to save.
    :param output_file: File path to save the DataFrame.
    """
    df.to_csv(output_file, index=True)

def fill_data(df):
    """
    Fills missing data using linear interpolation.
    
    :param df: DataFrame with missing data.
    :return: DataFrame with missing values filled.
    """
    df_filled = df.interpolate(method='linear', limit_direction='both', axis=0)
    return df_filled


def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file',
        type=str,
        default='../data/raw_data.csv',
        help='Path to the raw data file to process'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='../data/processed_data.csv', 
        help='Path to save the processed data'
    )
    return parser.parse_args()   

def main(input_file, output_file):
    """
    Main function to execute the data processing pipeline.
    
    :param input_file: Path to the input data directory.
    :param output_file: Path to the output CSV file.
    """
    df = process_energy_data(os.path.join(os.path.split(os.getcwd())[0], 'data'))
    df_clean = clean_data(df)
    df_processed = preprocess_data(df_clean)
    save_data(df_processed, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.output_file)