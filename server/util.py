import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(Suburb,distance,rooms):
    try:
        loc_index = __data_columns.index(Suburb.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = distance
    x[1] = rooms
    if loc_index>=0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global  __data_columns
    global __locations

    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk

    global __model
    if __model is None:
        with open('./artifacts/melbourne_home_prices_model.pickle', 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('Westmeadows',7, 3))
    print(get_estimated_price('Abbotsford',10, 2))
    print(get_estimated_price('Albert Park', 9, 3,)) # other location
    print(get_estimated_price('Windsor', 1000, 7))  # other location