import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Data manipulation
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from tqdm import tqdm
 
# Plotting and visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn models and utilities
from sklearn.gaussian_process.kernels import ConstantKernel as C

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer

# File handling
import os

# Creating a new respository to save study plots 
PATH_Plots = './plots_saved'
if not os.path.exists(PATH_Plots):
    os.makedirs(PATH_Plots)

# Creating a new respository to store new dfs
PATH_DFs = './new_dfs'
if not os.path.exists(PATH_DFs):
    os.makedirs(PATH_DFs)

# Creating a new respository to save models results
PATH_train_models = './models'
if not os.path.exists(PATH_train_models):
    os.makedirs(PATH_train_models)

def pre_process_data_original(data_train, data_test, df_path, transform_scale=False):
    """
    This function preprocesses the training and test datasets by extracting datetime features,
    adding bank holiday flags, handling missing values, and applying scaling transformations if required.
    """
 
    data_train = pd.get_dummies(data_train, columns=["store_and_fwd_flag"], prefix="store_and_fwd", drop_first=True)
    data_test = pd.get_dummies(data_test, columns=["store_and_fwd_flag"], prefix="store_and_fwd", drop_first=True)
        
    print(f"starting fix date for train")
    data_train['tpep_pickup_datetime'] = pd.to_datetime(data_train['tpep_pickup_datetime'])
    data_train['tpep_dropoff_datetime'] = pd.to_datetime(data_train['tpep_dropoff_datetime'])

    # Create separate features for pickup datetime
    data_train['pickup_month'] = data_train['tpep_pickup_datetime'].dt.month.astype(int)
    data_train['pickup_day'] = data_train['tpep_pickup_datetime'].dt.day.astype(int)
    data_train['pickup_hour'] = data_train['tpep_pickup_datetime'].dt.hour.astype(int)
    data_train['pickup_weekday'] = data_train['tpep_pickup_datetime'].dt.dayofweek.astype(int)

    # Create separate features for dropoff datetime
    data_train['dropoff_month'] = data_train['tpep_dropoff_datetime'].dt.month.astype(int)
    data_train['dropoff_day'] = data_train['tpep_dropoff_datetime'].dt.day.astype(int)
    data_train['dropoff_hour'] = data_train['tpep_dropoff_datetime'].dt.hour.astype(int)
    data_train['dropoff_weekday'] = data_train['tpep_dropoff_datetime'].dt.dayofweek.astype(int)

    # Define bank holidays in NYC
    bank_holidays = [
    "2024-01-01",  # New Year's Day
    "2024-01-15",  # Martin Luther King Jr. Day
    ]
    bank_holidays = pd.to_datetime(bank_holidays)

    # Check if each pickup and dropoff date is a bank holiday
    data_train['pickup_is_bank_holiday'] = data_train['tpep_pickup_datetime'].dt.date.isin(bank_holidays.date).astype(int)
    data_train['dropoff_is_bank_holiday'] = data_train['tpep_dropoff_datetime'].dt.date.isin(bank_holidays.date).astype(int)

    #data_train = data_train.sort_values(by='tpep_pickup_datetime')

    # Drop original datetime columns
    data_train = data_train.drop(columns=["tpep_pickup_datetime", "tpep_dropoff_datetime"])

    # Save the updated dataframe to CSV
    data_train.to_csv(f"{df_path}/updated_train_after_fix_date.csv", index=False)
    print(f"fix data for train is done")
    
    # Adjust the display options to show full rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_colwidth', None)  # Show full column width if needed
    pd.set_option('display.expand_frame_repr', False)  # Prevent wrapping to the next line
    
    print(f"starting fix date for test")
    data_test['tpep_pickup_datetime'] = pd.to_datetime(data_test['tpep_pickup_datetime'])
    data_test['tpep_dropoff_datetime'] = pd.to_datetime(data_test['tpep_dropoff_datetime'])

    # Create separate features for pickup datetime
    data_test['pickup_month'] = data_test['tpep_pickup_datetime'].dt.month.astype(int)
    data_test['pickup_day'] = data_test['tpep_pickup_datetime'].dt.day.astype(int)
    data_test['pickup_hour'] = data_test['tpep_pickup_datetime'].dt.hour.astype(int)
    data_test['pickup_weekday'] = data_test['tpep_pickup_datetime'].dt.dayofweek.astype(int)

    # Create separate features for dropoff datetime
    data_test['dropoff_month'] = data_test['tpep_dropoff_datetime'].dt.month.astype(int)
    data_test['dropoff_day'] = data_test['tpep_dropoff_datetime'].dt.day.astype(int)
    data_test['dropoff_hour'] = data_test['tpep_dropoff_datetime'].dt.hour.astype(int)
    data_test['dropoff_weekday'] = data_test['tpep_dropoff_datetime'].dt.dayofweek.astype(int)

    # Define bank holidays in NYC
    bank_holidays = [
    "2024-01-01",  # New Year's Day
    "2024-01-15",  # Martin Luther King Jr. Day
    ]
    bank_holidays = pd.to_datetime(bank_holidays)

    # Check if each pickup and dropoff date is a bank holiday
    data_test['pickup_is_bank_holiday'] = data_test['tpep_pickup_datetime'].dt.date.isin(bank_holidays.date).astype(int)
    data_test['dropoff_is_bank_holiday'] = data_test['tpep_dropoff_datetime'].dt.date.isin(bank_holidays.date).astype(int)

    # Drop original datetime columns
    data_test = data_test.drop(columns=["tpep_pickup_datetime", "tpep_dropoff_datetime"])

    # Save the updated dataframe to CSV
    data_test.to_csv(f"{df_path}/updated_test_after_fix_date.csv", index=False)
    print(f"fix data for test is done")

    if transform_scale == True:
        """ 
        If scaling is required, apply QuantileTransformer to the specified columns for normalization
        and MinMaxScaler for other columns.
        """
        
        print(f"computing transform scale")
        qt = QuantileTransformer(output_distribution='normal')

        scaler = MinMaxScaler()

        # The columns to be transformed
        cols = ['trip_distance', 'fare_amount', 'PU_location_lat', 'PU_location_lon', 'DO_location_lat', 'DO_location_lon']

        # The columns to be scaled
        scaled_cols = data_train.columns
        scaled_cols = scaled_cols.drop('tip_amount')  # Ensure 'tip_amount' is excluded
        print(scaled_cols)

        # Fit on training data, then transform both train and test data for PowerTransformer
        data_train[cols] = qt.fit_transform(data_train[cols])
        data_test[cols] = qt.transform(data_test[cols])

        # Fit on training data, then transform both train and test data for MinMaxScaler
        train_data_scaled = scaler.fit_transform(data_train[scaled_cols])
        test_data_scaled = scaler.transform(data_test[scaled_cols])

        # Replace the scaled numeric columns back into the original dataframes
        data_train[scaled_cols] = train_data_scaled
        data_test[scaled_cols] = test_data_scaled
        print(f"transform scale is done")
        
        data_train.to_csv(f"{df_path}/train_after_transformscale.csv", index=False)
        data_test.to_csv(f"{df_path}/test_after_transformscale.csv", index=False)
        
        return data_train, data_test
    
    return data_train, data_test

    
def drop_outliers(train_data):
    """
    This function removes rows from the training data, which are considered as outliers.
    """
    valid_passenger = (train_data.passenger_count > 0)
    valid_distance = (train_data.trip_distance < 30)
    data_new = train_data[valid_passenger & valid_distance]
    print(f"outliners are dropped")
    return data_new
