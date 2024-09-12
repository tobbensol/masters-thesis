import math

from traffic.core import Traffic


def get_takeoff_and_landing_directions(flights: Traffic) -> Traffic:
    for flight in flights:
        start_direction: float = flight.first('30 sec').data.get(['heading']).median().values[0]
        end_direction: float = flight.last('30 sec').data.get(['heading']).median().values[0]
        yield start_direction * math.pi / 180, end_direction * math.pi / 180
