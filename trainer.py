
from geopy.geocoders import MapBox
from geopy.distance import geodesic
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

import json

import os



mapbox_api_key = os.environ['MAPBOX_API_KEY']



openai_api_key = os.environ["OPENAI_API_KEY"]

API_SECRET = os.environ['API_SECRET']


geocoder = MapBox(api_key=mapbox_api_key)






def get_coordinates(address):
    mapbox_api_key = os.environ["mapbox_api_key"]
    geocoder = MapBox(api_key=mapbox_api_key)

    # Convert the address to the central zipcode
    location = geocoder.geocode(address)

    if location:
        latitude, longitude = location.latitude, location.longitude
        print(f"Confirmed:\nAddress: {address}\nCoordinates: {latitude}, {longitude}")
        return latitude, longitude
    else:
        raise ValueError("Could not retrieve location information from the address.")


def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).miles


def train(user_location):
    print("i reached train user location")
    count = 0
    print(count)
    try:
        # os.remove("faiss.pkl")
        # os.remove("training.index")
        print("Removed existing faiss.pkl and training.index")
    except FileNotFoundError:
        pass

        # Check there is data fetched from the database
    training_data_folders = list(Path("training/facts/").glob("**/latitude*,longitude*"))

    # Check there is data in the trainingData folder
    if len(training_data_folders) < 1:
        print("The folder training/facts should be populated with at least one subfolder.")
        return

    latitude, longitude = user_location
    user_coordinates = (latitude, longitude)

    data = []
    for folder in training_data_folders:
        folder_coordinates = folder.name.replace('latitude', '').replace('longitude', '').split(',')
        folder_latitude, folder_longitude = map(float, folder_coordinates)

        folder_coords = (folder_latitude, folder_longitude)
        distance = calculate_distance(user_coordinates, folder_coords)
        print(f" the distance between {user_coordinates} and {folder.name} is {distance}.")

        if distance < 100:
            count = count + 1
            print(f"Added {folder.name}'s contents to training data.")
            for json_file in folder.glob("*.json"):
                with open(json_file) as f:
                    data.extend(json.load(f))
                    print(f"  Added {json_file.name} to training data.")

    if count == 0:
        print("No relevant data found within 50 miles.")
        print(count)
        return count

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)

    docs = []
    for entry in data:
        address = entry.get('address', '')
        if address is not None:
            print(f"Address to split: {address}")
            docs.extend(text_splitter.split_text(address))

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    store = FAISS.from_texts(docs, embeddings)

    # faiss.write_index(store.index, "training.index")
    # store.index = None
    # print(dir(store))
    print(count)
    # with open("faiss.pkl", "wb") as f:
    #    pickle.dump(store, f)

    return count