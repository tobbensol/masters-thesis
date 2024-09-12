from typing import Callable, List

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
    filtered_flights: List[Flight] = filter(f, flights)
    filtered_traffic: Traffic = Traffic.from_flights(filtered_flights)

    return filtered_traffic


def complete_flight_filter(departure: str, arrival: str):
    def complete_flights(flight: Flight) -> Traffic:
        departure_airport = airports[ICAO_codes[departure]]
        arrival_airport = airports[ICAO_codes[arrival]]

        start: Flight = flight.first('5 sec').data.get(['longitude', 'latitude']).median().values
        end: Flight = flight.last('5 sec').data.get(['longitude', 'latitude']).median().values

        start_longitude, start_latitude = start
        end_longitude, end_latitude = end

        # just the value I found filtered out the values the best
        epsilon = 0.03
        return (abs(departure_airport.latitude - start_latitude) < epsilon) and \
            (abs(departure_airport.longitude - start_longitude) < epsilon) and \
            (abs(arrival_airport.latitude - end_latitude) < epsilon) and \
            (abs(arrival_airport.longitude - end_longitude) < epsilon)

    return complete_flights
