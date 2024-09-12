from typing import Callable
from functions.data_filtering import filter_flights

import pandas as pd
import pickle
import os.path

from traffic.core import Traffic, Flight
from datetime import datetime, timedelta
from pyopensky.trino import Trino

ICAO_codes = {"bergen": "ENBR",
              "oslo": "ENGM",
              "gatwick": "EGKK",
              "heathrow": "EGLL",
              "new york": "KJFK",
              "cape town": "FACT",
              "los angeles": "KLAX"}
query = Trino()


def get_data_range(origin: str, destination: str, start: datetime, stop: datetime) -> Traffic:
    path = f"data/{origin}-{destination}-{start.date()}-{stop.date()}.pkl"

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


def get_filtered_data_range(origin: str, destination: str, start: datetime, stop: datetime, filter: Callable[[Flight], bool]):
    # return if it exists
    path = f"data/{origin}-{destination}-{start.date()}-{stop.date()}-{filter.__name__}.pkl"
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    else: # get data if it doesnt
        unfiltered_flights = get_data_range(origin=origin, destination=destination, start=start, stop=stop)
    filtered_flights = filter_flights(filter, unfiltered_flights)

    # and save the result
    with open(path, "wb") as f:
        pickle.dump(filtered_flights, f)
    return filtered_flights
