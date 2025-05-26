import gudhi
import numpy as np
import pandas as pd
import pickle
import os.path

from datetime import datetime, timedelta
from typing import Callable, List, Tuple, Optional

import requests
from numpy.f2py.auxfuncs import throw_error
from scipy.cluster.hierarchy import linkage
from tqdm import tqdm
from traffic.core import Traffic, Flight
from pyopensky.trino import Trino
from traffic.data import opensky

from functions.data_filtering import filter_flights, ICAO_codes, large_gap_filter
from functions.data_processing import split_flights, flight_persistence, sublevelset_persistence, \
    sublevelset_heading_persistence
from functions.objects import PersistenceData

client_id = 'b72279bf-a268-4cf1-96bb-2f2e290349df'
query = Trino()

def get_filtered_data_range(flights: List[Flight], file_name, f: Callable[[Flight], bool], load_results: bool = True) -> Tuple[Optional[List[Flight]], str]:
    file_name = f"{f.__name__}/{file_name}"
    path = f"data/{file_name}"

    if os.path.isfile(path) and load_results:
        with open(path, "rb") as file:
            return pickle.load(file), file_name

    filtered_flights = filter_flights(f, flights)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    # and save the result
    with open(path, "wb") as file:
        pickle.dump(filtered_flights, file)
    return filtered_flights, file_name


def get_flight_persistence(flights: List[Flight], file_name: str, load_results: bool = True) -> Tuple[List[gudhi.simplex_tree.SimplexTree],List[np.ndarray], str]:
    file_name = f"persistence/{file_name}"
    path = f"data/{file_name}"
    if os.path.isfile(path) and load_results:
        with open(path, "rb") as file:
            trees, paths = pickle.load(file)
            return trees, paths, file_name

    trees, paths = flight_persistence(flights)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump((trees, paths), file)
    return trees, paths, file_name


def get_condensed_distance_matrix(trees: List[gudhi.simplex_tree.SimplexTree], file_name: str, load_results=True):
    file_name = f"distance_matrices/{file_name}"
    path = f"data/{file_name}"

    if os.path.isfile(path) and load_results:
        with open(path, "rb") as f:
            return pickle.load(f)

    condensed_distance_matrix = []
    for i in tqdm(range(len(trees))):
        for j in range(i + 1, len(trees)):
            pers_i = trees[i]
            pers_j = trees[j]
            dist = gudhi.bottleneck_distance(pers_i, pers_j, 0.0001)
            condensed_distance_matrix.append(dist)

    with open(path, "wb") as file:
        pickle.dump(condensed_distance_matrix, file)
    return condensed_distance_matrix


def get_flight_persistances(flights, file_name, load_results: bool = True) -> Tuple[PersistenceData, PersistenceData, PersistenceData, PersistenceData]:
    persistence_path = f"data/landings/Persistence/{file_name}.pkl"

    if os.path.isfile(persistence_path) and load_results:
        with open(persistence_path, "rb") as file:
            return pickle.load(file)

    LL_persistence, LL_paths = flight_persistence(flights)
    LL_data = PersistenceData(LL_persistence, LL_paths, "LL")

    A_persistence, A_paths = sublevelset_persistence(flights, "geoaltitude")
    A_data = PersistenceData(A_persistence, A_paths, "A")

    S_persistence, S_paths = sublevelset_persistence(flights, "groundspeed")
    S_data = PersistenceData(S_persistence, S_paths, "S")

    H_persistence, H_paths = sublevelset_heading_persistence(flights)
    H_data = PersistenceData(H_persistence, H_paths, "H")

    pers_objects = (LL_data, A_data, S_data, H_data)
    with open(persistence_path, "wb") as file:
        pickle.dump(pers_objects, file)
    return pers_objects


def flights_from_query(query, file_name: str, delta_time: pd.Timedelta = pd.Timedelta(minutes=15), load_results=True):
    flight_path = f"data/landings/Flights/{file_name}.pkl"
    label_path = f"data/landings/Labels/{file_name}.pkl"
    query_path = f"data/landings/Queries/{file_name}.pkl"

    if os.path.isfile(flight_path) and os.path.isfile(label_path) and load_results:
        with open(flight_path, "rb") as file:
            flight = pickle.load(file)
        with open(label_path, "rb") as file:
            labels = pickle.load(file)
        return flight, labels

    if query is None:
        if os.path.isfile(query_path):
            with open(query_path, "rb") as file:
                query = pickle.load(file)
        else:
            raise Exception("query file not found")
    else:
        with open(query_path, "wb") as file:
            pickle.dump(query, file)

    flights = []
    other_data = []
    for _, row in tqdm(query.iterrows(), total=query.shape[0]):
        # take at most 10 minutes before and 10 minutes after the landing or go-around
        start_time = row["time"] - delta_time
        stop_time = row["time"] + delta_time

        # fetch the data from OpenSky Network
        flights.append(
            opensky.history(
                start=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                stop=stop_time.strftime("%Y-%m-%d %H:%M:%S"),
                callsign=row["callsign"],
                return_flight=True,
            )
        )
        n_approaches = row["n_approaches"]
        wind_speed_knts = row["wind_speed_knts"]
        visibility_m = row["visibility_m"]
        temperature_deg = row["temperature_deg"]

        other_data.append([n_approaches, wind_speed_knts, visibility_m, temperature_deg])

    other_data = np.array(other_data)

    filtered_flight_data = [(f, d) for f, d in zip(flights, other_data) if large_gap_filter(f)]
    flights, other_data = zip(*filtered_flight_data)
    other_data = np.array(other_data)

    with open(flight_path, "wb") as file:
        pickle.dump(flights, file)
    with open(label_path, "wb") as file:
        pickle.dump(other_data, file)

    return flights, other_data


def get_data_range(origin: str, destination: str, start: datetime, stop: datetime, load_results: bool = True) -> Tuple[List[Flight], str]:
    file_name = f"{origin}-{destination}-{start.date()}-{stop.date()}.pkl"
    path = f"data/unfiltered/{file_name}"

    if os.path.isfile(path) and load_results:
        with open(path, "rb") as f:
            return pickle.load(f), file_name

    data = []
    days = (stop - start).days
    if days <= 0:
        raise ValueError("The stop date must be after the start date.")

    day_range = [start + timedelta(days=n) for n in range(days)]

    for date in tqdm(day_range, total=days):
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
    flights.data = flights.data.dropna(subset=["heading", "latitude", "longitude"])
    final_list = split_flights(flights)

    # cache result
    with open(path, "wb") as file:
        pickle.dump(final_list, file)
    return final_list, file_name


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