from typing import Callable, Iterable

from traffic.core import Traffic, Flight
from traffic.data import airports

ICAO_codes = {"bergen": "ENBR",
              "oslo": "ENGM",
              "gatwick": "EGKK",
              "heathrow": "EGLL",
              "new york": "KJFK",
              "cape town": "FACT",
              "los angeles": "KLAX"}


def filter_flights(f: Callable[[Flight], bool], flights: Traffic) -> Traffic:
    filtered_flights: Iterable[Flight] = filter(f, flights)
    filtered_traffic: Traffic = Traffic.from_flights(filtered_flights)

    return filtered_traffic


def complete_flight_filter(departure: str, arrival: str) -> Callable[[Flight], bool]:
    # all filters must have this signature
    def complete_flights(flight: Flight) -> bool:
        departure_airport = airports[ICAO_codes[departure]]
        arrival_airport = airports[ICAO_codes[arrival]]

        start_longitude, start_latitude = flight.first('5 sec').data.get(['longitude', 'latitude']).median().values
        end_longitude, end_latitude = flight.last('5 sec').data.get(['longitude', 'latitude']).median().values

        # just the value I found filtered out the values the best
        epsilon = 0.03
        return (abs(departure_airport.latitude - start_latitude) < epsilon) and \
            (abs(departure_airport.longitude - start_longitude) < epsilon) and \
            (abs(arrival_airport.latitude - end_latitude) < epsilon) and \
            (abs(arrival_airport.longitude - end_longitude) < epsilon)

    return complete_flights
