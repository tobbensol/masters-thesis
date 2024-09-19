import pandas as pd
import pickle
import os.path

from datetime import datetime, timedelta
from typing import Callable

from traffic.core import Traffic, Flight
from pyopensky.trino import Trino

from functions.data_filtering import filter_flights

ICAO_codes = {"bergen": "ENBR",
              "oslo": "ENGM",
              "gatwick": "EGKK",
              "heathrow": "EGLL",
              "new york": "KJFK",
              "cape town": "FACT",
              "los angeles": "KLAX"}
query = Trino()


def get_data_range(origin: str, destination: str, start: datetime, stop: datetime) -> Traffic:
    path = f"data/unfiltered/{origin}-{destination}-{start.date()}-{stop.date()}.pkl"

    if os.path.isfile(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    data = []
    days = (stop - start).days
    if days <= 0:
        print("stop date has to be after the start date")
        return

    for date in (start + timedelta(days=n) for n in range(days)):
        result = query.history(
            start=date,
            stop=date + timedelta(days=1),
            departure_airport=ICAO_codes[origin],
            arrival_airport=ICAO_codes[destination])
        if result is not None:
            data.append(result.copy())

    # combine data
    result = pd.concat(data, axis="rows", ignore_index=True)

    # make flight object
    final = result.rename(columns={'time': 'timestamp', 'lat': 'latitude', 'lon': 'longitude'})
    flights = Traffic(final)

    # cache result
    with open(path, "wb") as f:
        pickle.dump(flights, f)
    return flights


def get_filtered_data_range(origin: str, destination: str, start: datetime, stop: datetime, f: Callable[[Flight], bool]):
    path = f"data/{f.__name__}/{origin}-{destination}-{start.date()}-{stop.date()}.pkl"
    if os.path.isfile(path):
        with open(path, "rb") as file:
            return pickle.load(file)
    else: # get data if it doesn't exist yet
        unfiltered_flights = get_data_range(origin=origin, destination=destination, start=start, stop=stop)
    filtered_flights = filter_flights(f, unfiltered_flights)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    # and save the result
    with open(path, "wb") as file:
        pickle.dump(filtered_flights, file)
    return filtered_flights
