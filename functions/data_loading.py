import gudhi
import pandas as pd
import pickle
import os.path

from datetime import datetime, timedelta
from typing import Callable, List, Tuple, Optional

import requests
from Crypto.SelfTest.Cipher.test_CFB import file_name
from tqdm import tqdm
from traffic.core import Traffic, Flight
from pyopensky.trino import Trino

from functions.data_filtering import filter_flights
from functions.data_processing import generate_alpha_tree

client_id = 'b72279bf-a268-4cf1-96bb-2f2e290349df'

ICAO_codes = {"bergen": "ENBR",
              "oslo": "ENGM",
              "gatwick": "EGKK",
              "heathrow": "EGLL",
              "new york": "KJFK",
              "cape town": "FACT",
              "los angeles": "KLAX"}
query = Trino()


def get_data_range(origin: str, destination: str, start: datetime, stop: datetime) -> Tuple[Traffic, str]:
    file_name = f"{origin}-{destination}-{start.date()}-{stop.date()}.pkl"
    path = f"data/unfiltered/{file_name}"

    if os.path.isfile(path):
        with open(path, "rb") as f:
            return pickle.load(f), file_name

    data = []
    days = (stop - start).days
    if days <= 0:
        raise ValueError("The stop date must be after the start date.")

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
    return flights, file_name


def get_filtered_data_range(traffic, file_name, f: Callable[[Flight], bool]) -> Tuple[Optional[Traffic], str]:
    file_name = f"{f.__name__}/{file_name}"
    path = f"data/{file_name}"
    if os.path.isfile(path):
        with open(path, "rb") as file:
            return pickle.load(file), file_name
    filtered_flights = filter_flights(f, traffic)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    # and save the result
    with open(path, "wb") as file:
        pickle.dump(filtered_flights, file)
    return filtered_flights, file_name


def get_flight_persistence(traffic: Traffic, file_name) -> Tuple[List[gudhi.simplex_tree.SimplexTree], str]:
    file_name = f"persistence/{file_name}"
    path = f"data/{file_name}"
    if os.path.isfile(path):
        with open(path, "rb") as file:
            return pickle.load(file), file_name

    to_save = []
    for i in tqdm(range(len(traffic))):
        flight = traffic[i]
        tree = generate_alpha_tree(flight)
        to_save.append(tree)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(to_save, file)
    return to_save, file_name


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
        'elements': 'wind_from_direction,wind_speed',
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
            wind_direction = None
            wind_speed = None

            # Iterate over observations to extract both wind direction and wind speed
            for observation in item.get('observations', []):
                if observation['elementId'] == 'wind_from_direction':
                    wind_direction = observation['value']
                elif observation['elementId'] == 'wind_speed':
                    wind_speed = observation['value']

            # Only append if both wind direction and speed are available
            if wind_direction is not None and wind_speed is not None:
                observations.append({
                    'source': item['sourceId'],
                    'time': item['referenceTime'],
                    'wind_direction': wind_direction,
                    'wind_speed': wind_speed
                })

        # Convert observations to a pandas DataFrame
        return pd.DataFrame(observations)

    else:
        print(f"Error: {response.status_code}, {response.text}")