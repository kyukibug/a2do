import os
import time
import json

import requests
import numpy as np
import faiss
import openai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")

# this is location of ann arbor
LOCATION = "42.2808,-83.7430"

# radius in meters of search
RADIUS = 20000

# types of places to search for
PLACE_TYPES = ["restaurant", "cafe", "park", "library", "museum"]

PLACES_FILE = "places_data.json"
EMBEDDINGS_FILE = "embeddings.npy"
FAISS_INDEX_FILE = "faiss_index.bin"

if not API_KEY:
    raise ValueError(
        "No API key found. Set the API_KEY environment variable in your .env file."
    )


def fetch_places():
    all_places = []
    for place_type in PLACE_TYPES:
        print(f"Fetching places of type: {place_type}")
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "key": API_KEY,
            "location": LOCATION,
            "radius": RADIUS,
            "type": place_type,
        }
        while True:
            response = requests.get(url, params=params)
            if response.status_code != 200:
                print(f"Error fetching data: {response.status_code}")
                break
            data = response.json()
            all_places.extend(data.get("results", []))
            next_page_token = data.get("next_page_token")
            if next_page_token:
                # we will sleep as per google's recommendation
                time.sleep(2)
                params["pagetoken"] = next_page_token
            else:
                break

    # remove duplicates using place_id
    unique_places = {place["place_id"]: place for place in all_places}
    return list(unique_places.values())


def save_places_data():
    if not os.path.exists(PLACES_FILE):
        places = fetch_places()
        with open(PLACES_FILE, "w") as f:
            json.dump(places, f)
        print(f"Saved {len(places)} places to {PLACES_FILE}")
    else:
        print(f"{PLACES_FILE} already exists. Skipping fetch.")


save_places_data()


def load_places_data():
    with open(PLACES_FILE, "r") as f:
        places = json.load(f)
    return places


places = load_places_data()
