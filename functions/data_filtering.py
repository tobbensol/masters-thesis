from traffic.core import Traffic, Flight
from traffic.data import airports

ICAO_codes = {"bergen": "ENBR",
              "oslo": "ENGM",
              "gatwick":"EGKK",
              "heathrow":"EGLL",
              "new york": "KJFK",
              "cape town":"FACT",
              "los angeles": "KLAX"}


def get_complete_flights(flights: Traffic, fro: str, to: str) -> Traffic:
    for flight in flights:
        departure = airports[ICAO_codes[fro]]
        arrival = airports[ICAO_codes[to]]

        start: Flight = flight.first('2 min').data.get(['longitude', 'latitude']).median().values
        end: Flight = flight.last('2 min').data.get(['longitude', 'latitude']).median().values

        start_longitude, start_latitude = start
        end_longitude, end_latitude = end

        if (abs(departure.latitude - start_latitude) < 0.5) and (abs(departure.longitude - start_longitude) < 0.5) and \
                (abs(arrival.latitude - end_latitude) < 0.5) and (abs(arrival.longitude - end_longitude) < 0.5):
            yield flight

