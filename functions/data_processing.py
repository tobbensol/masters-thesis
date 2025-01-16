import math

import gudhi
import numpy
import pandas as pd
import numpy as np

from datetime import datetime, timedelta

from gudhi.alpha_complex import AlphaComplex
from tqdm import tqdm
from traffic.core import Traffic, Flight
from typing import Tuple, List


def get_takeoff_and_landing_directions(flights: Traffic) -> Tuple[datetime, datetime, float, float]:
    for flight in flights:
        start_direction, start_time = flight.first('30 sec').data.get(['heading', 'timestamp']).median().values
        end_direction, end_time = flight.last('30 sec').data.get(['heading', 'timestamp']).median().values
        yield start_time, end_time, start_direction * np.pi / 180, end_direction * np.pi / 180

def get_date(flights: Traffic) -> datetime:
    for flight in flights:
        timestamp: datetime = flight.first('30 sec').data.get(['timestamp']).median().values[0]
        yield timestamp

def generate_alpha_tree(flight: Flight) -> gudhi.simplex_tree.SimplexTree:
    points = flight.data[['latitude', 'longitude']].dropna(axis="rows").to_numpy()
    alpha_complex: gudhi.alpha_complex = AlphaComplex(points=points)
    tree: gudhi.simplex_tree.SimplexTree = alpha_complex.create_simplex_tree()
    tree.compute_persistence()
    return tree

def flight_pers(flights):
    to_save = []
    for i in tqdm(range(len(flights))):
        flight = flights[i]
        tree = generate_alpha_tree(flight)
        to_save.append(tree)
    return to_save


def remove_outliers(flight: Flight) -> Flight:
    """
    this doesn't work for flights over the pacific ocean, would have to import some library for that
    :param flight: The flight you would like to remove outliars from
    :return: The same flight without outliers
    """
    df = flight.data.copy()

    # Calculate differences between consecutive latitude and longitude values
    lat_diff = np.diff(df['latitude'])
    lon_diff = np.diff(df['longitude'])

    # Approximate distance using Euclidean formula on lat/lon changes
    approx_distances = np.sqrt(lat_diff ** 2 + lon_diff ** 2)

    # Insert NaN at the beginning to align with DataFrame length
    approx_distances = np.insert(approx_distances, 0, np.nan)
    df['approx_distance'] = approx_distances

    # Define thresholds for approximate distances
    min_distance_threshold = 0.0001  # Minimum allowable distance in degrees
    max_distance_threshold = 0.005  # Maximum allowable distance in degrees

    # Filter out rows where distance is either too small (likely duplicate) or too large (likely outlier)
    df_cleaned = df[
        ((df['approx_distance'] >= min_distance_threshold) & (df['approx_distance'] <= max_distance_threshold)) |
        (df['approx_distance'].isna())  # Keep the first point
        ]
    new_flight = Flight(df_cleaned)
    return new_flight

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