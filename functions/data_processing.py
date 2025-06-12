import gudhi
import numpy
import pandas as pd
import numpy as np

from datetime import datetime, timedelta

from numpy import ndarray
from scipy.signal import savgol_filter
from sklearn.cluster import DBSCAN
from gudhi.alpha_complex import AlphaComplex
from tqdm import tqdm
from traffic.core import Traffic, Flight
from typing import Tuple, List, Callable


def get_takeoff_and_landing_directions(flights: Traffic) -> Tuple[datetime, datetime, float, float]:
    for flight in flights:
        start_direction, start_time = flight.first('30 sec').data.get(['heading', 'timestamp']).median().values
        end_direction, end_time = flight.last('30 sec').data.get(['heading', 'timestamp']).median().values
        yield start_time, end_time, start_direction * np.pi / 180, end_direction * np.pi / 180


def get_date(flights: Traffic) -> datetime:
    for flight in flights:
        timestamp: datetime = flight.first('30 sec').data.get(['timestamp']).median().values[0]
        yield timestamp

def second_dim_persistence(flights, columns: List[str]) -> Tuple[List[gudhi.simplex_tree.SimplexTree], List[numpy.ndarray]]:
    trees = []
    paths = []
    for i in tqdm(range(len(flights))):
        f = (lambda x: remove_outliers_dbscan(x, min_samples=25, eps = 0.05))

        data = get_columns_timestamp_index(flights[i], columns)
        data = clean_flight_data(data, drop_duplicates=True, f=f)

        alpha_complex: gudhi.alpha_complex = AlphaComplex(points=data)
        tree: gudhi.simplex_tree.SimplexTree = alpha_complex.create_simplex_tree()
        tree.compute_persistence()

        trees.append(tree.persistence_intervals_in_dimension(1))
        paths.append(data)
    return trees, paths

def sublevelset_persistence(flights: List[Flight], parameter: str):
    trees = []
    paths = []
    for i in tqdm(range(len(flights))):
        f = (lambda x: remove_outliers_dbscan(x, eps=1000, min_samples=75))

        data = get_columns_timestamp_index(flights[i], [parameter])
        data = clean_flight_data(data, f = f)
        data = data.reshape(data.shape[0] * data.shape[1])
        path = np.column_stack((np.arange(len(data)), data))

        st = build_sublevelset_filtration(data)
        st.compute_persistence()

        tree = st.persistence_intervals_in_dimension(0)
        tree[tree.shape[0]-1, tree.shape[1]-1] = max(data)

        trees.append(tree)
        paths.append(path)
    return trees, paths


def sublevelset_heading_persistence(flights: List[Flight]):
    trees = []
    paths = []
    for i in tqdm(range(len(flights))):
        f = None
        data = get_columns_timestamp_index(flights[i], ["track"])
        data["track"] = np.unwrap(np.deg2rad(data["track"]), period=2 * np.pi, discont=np.pi)

        data = clean_flight_data(data, f = f)
        data = data.reshape(data.shape[0] * data.shape[1])
        path = np.column_stack((np.arange(len(data)), data))

        st = build_sublevelset_filtration(data)
        st.compute_persistence()

        tree = st.persistence_intervals_in_dimension(0)
        tree[tree.shape[0]-1, tree.shape[1]-1] = max(data)

        trees.append(tree)
        paths.append(path)
    return trees, paths


def get_flight_statistics(flights: List[Flight]):
    statistics = []
    for flight in flights:
        data = flight.data.copy(deep=True)

        unwrapped = np.unwrap(np.deg2rad(data["track"]), period=2 * np.pi, discont=np.pi)
        max_unwrapped = unwrapped.max()
        min_unwrapped = unwrapped.min()
        diff_unwrapped = max_unwrapped - min_unwrapped

        max_velocity = data["groundspeed"].max()
        min_velocity = data["groundspeed"].min()
        mean_velocity = data["groundspeed"].mean()
        max_heading = data["vertical_rate"].max()
        min_heading = data["vertical_rate"].min()
        mean_heading = data["vertical_rate"].mean()
        longitude_diff = data["longitude"].max() - data["longitude"].min()
        latitude_diff = data["latitude"].max() - data["latitude"].min()

        statistics.append(
            [max_unwrapped, min_unwrapped, diff_unwrapped, max_velocity, min_velocity, mean_velocity, max_heading,
             min_heading, mean_heading, longitude_diff, latitude_diff])

    statistics = np.array(statistics)
    return statistics


def clean_flight_data(data: np.ndarray[float], drop_duplicates: bool = False, f: Callable[[np.ndarray[float]], np.ndarray[bool]]=None) -> np.ndarray:
    if drop_duplicates:
        data = data.drop_duplicates()

    if f is not None:
        inliers = f(data)
        data = data[inliers]

    data = data.resample("5s").mean()
    data = data.interpolate("time")
    cols = []
    for i in data.columns:
        col = data[i].to_numpy()
        cols.append(savgol_filter(x=col, window_length=min(len(data)-1, 25), polyorder=2))
    data = np.array(cols).T
    return data

def get_columns_timestamp_index(flight: Flight, columns: List[str]):
    data = flight.data[(columns+['timestamp'])].copy().dropna(axis="rows")
    data = data.set_index(pd.DatetimeIndex(data['timestamp']))
    return data[columns]

def remove_outliers_z_score(points, threshold=3):
    # Compute z-scores for the data
    z_scores = np.abs((points - np.mean(points, axis=0)) / np.std(points, axis=0))

    inliers = np.all(z_scores <= threshold, axis=1)
    return inliers


def remove_outliers_dbscan(points, min_samples: int, eps: float) -> ndarray[bool]:
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)

    inliers = labels >= 0
    return inliers

def build_sublevelset_filtration(Y):
    """
    Y: array-like
        Array of function values
    """
    st = gudhi.SimplexTree()
    for i in range(len(Y)):
        # 0-simplices
        st.insert([i], filtration=Y[i])

        if i < len(Y) - 1:
            # 1-simplices
            st.insert([i, i + 1], filtration=max(Y[i], Y[i + 1]))

    return st

def split_flights(traffic: Traffic, threshold: timedelta = timedelta(seconds=60)) -> List[Flight]:
    flights = []
    for flight in traffic:
        # Calculate time differences between consecutive rows
        time_diffs = np.diff(flight.data["timestamp"])

        # Identify indices where the time gap exceeds the threshold
        split_indices = np.where(time_diffs > threshold)[0]

        # Split the DataFrame into segments
        dfs = []
        start_idx = 0
        for split_idx in split_indices:
            # Add the segment up to the split point
            dfs.append(flight.data.iloc[start_idx: split_idx + 1])
            start_idx = split_idx + 1
        # Add the final segment
        dfs.append(flight.data.iloc[start_idx:])

        # Convert each DataFrame segment into a Flight object
        for df in dfs:
            flights.append(Flight(df))

    return flights

def prepare_wind_data(df: pd.DataFrame):
    df = df.copy()

    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    df = df.loc[df["wind_direction"] != 0]

    df["wind_direction"] = np.unwrap(np.deg2rad(df["wind_direction"]), period=2 * np.pi, discont=np.pi)

    df = df.resample("s").mean(numeric_only=True)
    df.interpolate(inplace=True)

    df["wind_direction"] = df["wind_direction"] % (2 * np.pi)

    df["x"] = np.sin(df["wind_direction"])
    df["y"] = np.cos(df["wind_direction"])

    df["x_scaled"] = df["x"] * df["wind_speed"]
    df["y_scaled"] = df["y"] * df["wind_speed"]

    return df