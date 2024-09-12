import pandas as pd
import pickle
import os.path

from traffic.core import Traffic
from datetime import datetime, timedelta
from pyopensky.trino import Trino


ICAO_codes = {"bergen": "ENBR",
              "oslo": "ENGM",
              "gatwick":"EGKK",
              "heathrow":"EGLL",
              "new york": "KJFK",
              "cape town":"FACT",
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
                            stop =date + timedelta(days=1),
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
    pickle.dump(flights, open(path, 'wb'))
    return flights
