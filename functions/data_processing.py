import math

import gudhi
import pandas as pd
import numpy as np

from datetime import datetime

from gudhi.alpha_complex import AlphaComplex
from traffic.core import Traffic, Flight
from typing import Tuple



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
    points = flight.data[['latitude', 'longitude']]
    alpha_complex: gudhi.alpha_complex = AlphaComplex(points=points)
    tree: gudhi.simplex_tree.SimplexTree = alpha_complex.create_simplex_tree()
    tree.compute_persistence()
    return tree


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