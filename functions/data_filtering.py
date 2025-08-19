from typing import Callable, Sequence, Iterable, List

import pandas as pd
from traffic.core import Traffic, Flight
from traffic.data import airports

ICAO_codes = {"bergen": "ENBR",
              "oslo": "ENGM",
              "gatwick": "EGKK",
              "heathrow": "EGLL",
              "new york": "KJFK",
              "cape town": "FACT",
              "los angeles": "KLAX",
              "amsterdam": "EHAM",
              "mallorca": "LEPA",
              "barcelona": "LEBL",
              "frankfurt":"EDDF",
              "berlin":"EDDB"}


def filter_flights(f: Callable[[Flight], bool], flights: List[Flight]) -> List[Flight]:
    filtered_flights: Iterable[Flight] = filter(f, flights)
    return list(filtered_flights)


def complete_flight_filter(departure: str, arrival: str, epsilon: float = 0.03) -> Callable[[Flight], bool]:
    # all filters must have this signature
    def complete_flights(flight: Flight) -> bool:
        departure_airport = airports[ICAO_codes[departure]]
        arrival_airport = airports[ICAO_codes[arrival]]

        start_longitude, start_latitude = flight.first('5 sec').data.get(['Longitude', 'Latitude']).median().values
        end_longitude, end_latitude = flight.last('5 sec').data.get(['Longitude', 'Latitude']).median().values

        if pd.isna(start_longitude) or pd.isna(start_latitude) or pd.isna(end_longitude) or pd.isna(end_latitude):
            return False

        return (abs(departure_airport.latitude - start_latitude) < epsilon) and \
            (abs(departure_airport.longitude - start_longitude) < epsilon) and \
            (abs(arrival_airport.latitude - end_latitude) < epsilon) and \
            (abs(arrival_airport.longitude - end_longitude) < epsilon)

    return complete_flights


def filter_by_bools(bools: Sequence[bool]) -> Callable[[Flight], bool]:
    index = 0
    def bool_filter(flight: Flight) -> bool:
        nonlocal index
        value = bools[index]
        index += 1
        return value
    return bool_filter


def large_gap_filter(flight: Flight):
    """
    Filters out flights that have more than 30 secs of missing data.
    :param flight: The flight
    :return: True if the flight has no gaps longer than 30 seconds, False otherwise
    """
    indexes = flight.data[["longitude", "latitude", "geoaltitude"]].dropna(axis="rows").drop_duplicates().index
    timestamps = flight.data.iloc[indexes]["timestamp"].to_numpy()

    time_gap = pd.Series(timestamps, index=indexes).diff()
    if any(time_gap > pd.Timedelta(seconds=45)):
        return False
    else:
        return len(timestamps) > 250