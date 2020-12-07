import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer  # , StandardScaler


def get_transformed_california_data():
    """
    Get and transform California housing data for model.
    Parameters:
        None
    Returns:
        X_train - Transformed California housing training data
        X_test - Transformed California housing testing data
        y_train - Transformed California housing training values
        y_test - California housing test values
    """
    df = fetch_california_housing(as_frame=True)
    X_train, X_test, y_train, y_test = transform_california_data(df)
    return X_train, X_test, y_train, y_test


def transform_california_data(df):
    df = log_features(df)
    df = cut_features(df)
    df = add_features(df)
    df = normal_transform(df)
    df = remove_features(df)
    X_train, X_test, y_train, y_test = get_train_test(df)
    return X_train, X_test, y_train, y_test


def log_features(df):
    df["MedInc"] = np.log1p(df['MedInc'])
    df["AveRooms"] = np.log1p(df['AveRooms'])
    df["AveBedrms"] = np.log1p(df['AveBedrms'])
    df["Population"] = np.log1p(df['Population'])
    df["AveOccup"] = np.log1p(df['AveOccup'])
    df["MedHouseVal"] = np.log1p(df['MedHouseVal'])
    return df


def cut_features(df):
    df = df.drop(df[(df['HouseAge'] > 50) | (df['MedHouseVal'] > 1.75)].index)
    return df


def add_features(df):
    df['AveBedrmsPerRoom'] = df['AveBedrms'] / df['AveRooms']
    df['EstHouses'] = df['Population'] / df['AveOccup']
    df = add_geo_features(df)


def add_geo_features(df):
    town_coords = np.array([[37.4613, -122.1997],
                            [34.0195, -118.4912],
                            [34.0736, -118.4004],
                            [37.4419, -122.1430],
                            [37.3852, -122.1141],
                            [37.9624, -122.5550],
                            [37.3841, -122.2352],
                            [37.3478, -122.1008],
                            [33.6189, -117.9298],
                            [33.6061, -117.8912],
                            [32.7157, -117.1611]])
    coords = np.asarray(df[['Latitude', 'Longitude']])
    town_dists = np.asarray([np.sqrt((coords[:, 0] - town_coord[0])**2 +
                                     (coords[:, 1] - town_coord[1])**2)
                             for town_coord in town_coords])
    dist_to_town = np.min(town_dists, axis=0)
    df['DistToTown'] = dist_to_town
    return df


def normal_transform(df):
    qt = QuantileTransformer(output_distribution="normal")
    df = pd.DataFrame(qt.fit_transform(df), columns=df.columns)
    return df


def remove_features(df):
    remove = ['AveRooms', 'AveBedrms', 'Latitude',
              'Longitude', 'Population', 'HouseAge']
    df = df.drop(remove, axis=1, inplace=True)
    return df


def get_train_test(df):
    X_features = ['MedInc', 'AveOccup', 'AveBedrmsPerRoom',
                  'AveAddRooms', 'EstHouses', 'DistToTown']
    y_features = ['MedHouseVal']
    X = df[X_features]
    y = df[y_features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test