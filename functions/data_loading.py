import pandas as pd
import pickle
import os.path

from datetime import datetime, timedelta
from typing import Callable

import requests
from traffic.core import Traffic, Flight
from pyopensky.trino import Trino

from functions.data_filtering import filter_flights

client_id = 'b72279bf-a268-4cf1-96bb-2f2e290349df'

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
    else: # get data if it doesnt
        unfiltered_flights = get_data_range(origin=origin, destination=destination, start=start, stop=stop)
    filtered_flights = filter_flights(f, unfiltered_flights)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    # and save the result
    with open(path, "wb") as file:
        pickle.dump(filtered_flights, file)
    return filtered_flights


def get_weather_station_id(name: str) -> str:
    base_url = 'https://frost.met.no/sources/v0.jsonld'

    params = {
        'name': name,  # Search by names
    }
    response = requests.get(base_url, params=params, auth=(client_id, ''))
    if response.status_code == 200:
        data = response.json()

        # Extract station IDs and names
        for station in data.get('data', []):
            return station['id']
    else:
        return f"Error: {response.status_code}, {response.text}"


def get_wind_direction(name: str) -> pd.DataFrame:
    base_url = 'https://frost.met.no/observations/v0.jsonld'
    weather_station_id = get_weather_station_id(name)
    params = {
        'sources': weather_station_id,
        'elements': 'wind_from_direction',
        'referencetime': '2023-01-01/2024-01-01',
    }

    # Make the request with authentication
    response = requests.get(base_url, params=params, auth=(client_id, ''))

    # Check for successful request
    if response.status_code == 200:
        data = response.json()

        # Process the response to extract the relevant data
        observations = []

        # Extract observations from the response
        for item in data.get('data', []):
            for observation in item.get('observations', []):
                observations.append({
                    'source': item['sourceId'],
                    'time': item['referenceTime'],
                    'wind_direction': observation['value']
                })

        return pd.DataFrame(observations)

    else:
        print(f"Error: {response.status_code}, {response.text}")