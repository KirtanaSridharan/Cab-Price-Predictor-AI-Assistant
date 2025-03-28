import argparse
import math
import os
import random

import pandas as pd
import xgboost as xgb

from hypertune import HyperTune

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from google.cloud import storage

# Sample 40% of the whole dataset
SAMPLE_PROB = 0.4

# Set the random seed to get deterministic sampling results
random.seed(42)

# Google Cloud Storage bucket name
GCP_BUCKET = 'cab-fare-project-ks'

# Training file name
TRAIN_FILE = 'dataset/train.csv'

# Landmark Distances
jfk_coordinates = (40.6413, -73.7781)
lga_coordinates = (40.7769, -73.8740)
times_sq_coordinates = (40.7580, -73.9855)
empire_state_coordinates = (40.7484, -73.9857)
central_park_coordinates = (40.7829, -73.9654)

# Rush hours
rush_hours = [7, 8, 9, 16, 17, 18]


# Method to calculate the haversine distance between two coordinates
def haversine_distance(source, destination):
    source_lat, source_lon = source
    dest_lat, dest_lon = destination
    
    # Radius of the Earth in kilometers
    earth_radius = 6371

    dlat = math.radians(dest_lat - source_lat)
    dlon = math.radians(dest_lon - source_lon)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(source_lat)) * math.cos(
        math.radians(dest_lat)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = earth_radius * c

    return distance

# Add features to the DataFrame
def add_features(df):
    # Haversine distance feature
    df['haversine_distance'] = df.apply(
        lambda row: haversine_distance(
            (row['pickup_latitude'], row['pickup_longitude']),
            (row['dropoff_latitude'], row['dropoff_longitude'])
        ),
        axis=1
    )

    # Temporal features
    df['year'] = df['pickup_datetime'].dt.year
    df['month'] = df['pickup_datetime'].dt.month
    df['weekday'] = df['pickup_datetime'].dt.weekday
    df['hour'] = df['pickup_datetime'].dt.hour

    # Delta diff features
    df['diff_lat'] = df['pickup_latitude'] - df['dropoff_latitude']
    df['diff_lon'] = df['pickup_longitude'] - df['dropoff_longitude']

    # Landmark distance features
    df['pickup_from_jfk'] = df.apply(
        lambda row: haversine_distance((row['pickup_latitude'], row['pickup_longitude']), jfk_coordinates),
        axis=1
    )
    df['pickup_from_lga'] = df.apply(
        lambda row: haversine_distance((row['pickup_latitude'], row['pickup_longitude']), lga_coordinates),
        axis=1
    )
    df['pickup_from_times_sq'] = df.apply(
        lambda row: haversine_distance((row['pickup_latitude'], row['pickup_longitude']), times_sq_coordinates),
        axis=1
    )
    df['pickup_from_empire_state'] = df.apply(
        lambda row: haversine_distance((row['pickup_latitude'], row['pickup_longitude']), empire_state_coordinates),
        axis=1
    )
    df['pickup_from_central_park'] = df.apply(
        lambda row: haversine_distance((row['pickup_latitude'], row['pickup_longitude']), central_park_coordinates),
        axis=1
    )

    # Passenger count (assume one passenger if not provided)
    if 'passenger_count' not in df.columns:
        df['passenger_count'] = 1
    
    # If in rush hour (7-9 AM, 4-6 PM)
    df['is_rush_hour'] = df['hour'].isin(rush_hours).astype(int)

    return df

# Transform the training dataset
def transform_train_data(df):
    df = add_features(df)

    # Drop unnecessary columns to avoid noise in the model
    df = df.drop([
        'key',
        'pickup_datetime',
        'pickup_latitude',
        'pickup_longitude',
        'dropoff_latitude',
        'dropoff_longitude'
    ], axis=1, errors='ignore')

    # Drop outliers for training (99.9 percentile cap for fare amount)
    fare_cap = df['fare_amount'].quantile(0.999)
    df = df[df['fare_amount'] < fare_cap]

    return df

if __name__ == '__main__':
    # Download the training data from Google Cloud Storage bucket
    gcp_bucket = storage.Client().bucket(GCP_BUCKET)
    gcp_bucket.blob(TRAIN_FILE).download_to_filename('train.csv')

    # Load the training data and transform it
    original_train_data = pd.read_csv(
        'train.csv',
        parse_dates=["pickup_datetime"],
        skiprows=lambda i: i > 0 and random.random() > SAMPLE_PROB
    )
    df_train = transform_train_data(original_train_data)

    # Define target and features
    X = df_train.drop(['fare_amount'], axis=1, errors='ignore')
    Y = df_train['fare_amount']

    # Split the data into training and validation sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    # Parse hyperparameters args from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_depth', default=6, type=int)
    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--subsample', default=1.0, type=float)
    parser.add_argument('--n_estimators', default=100, type=int)
    args = parser.parse_args()
    params = {
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'subsample': args.subsample,
        'n_estimators': args.n_estimators,
    }

    # Create DMatrix for XGBoost from DataFrames
    d_matrix_train = xgb.DMatrix(X_train, Y_test)
    d_matrix_eval = xgb.DMatrix(X_test)
    model = xgb.train(params, d_matrix_train)
    Y_pred = model.predict(d_matrix_eval)
    rmse = math.sqrt(mean_squared_error(Y_test, Y_pred))
    print('RMSE: {:.3f}'.format(rmse))

    # Report the score to hypertune to tune the hyperparameters in the next trial
    hypert = HyperTune()
    hypert.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='rmse_cab_fare', metric_value=rmse)

    # Get the job name and trial ID from environment variables
    VERTEX_JOB_ID = os.environ['CLOUD_ML_JOB_ID']
    VERTEX_TRIAL_ID = os.environ['CLOUD_ML_TRIAL_ID']

    # Save the model to Google Cloud Storage
    model_name = 'cab_pred_model.bst'
    model.save_model(model_name)
    blob = gcp_bucket.blob(f'{VERTEX_JOB_ID}/{VERTEX_TRIAL_ID}_rmse{rmse:.3f}_{model_name}')
    blob.upload_from_filename(model_name)
