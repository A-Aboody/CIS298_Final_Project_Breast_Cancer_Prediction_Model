import pandas as pd

def load_data():
    raw_data = pd.read_csv("Data/data.csv")

    processed_data = raw_data.drop(['Unnamed: 32', 'id'], axis=1)

    processed_data['diagnosis'] = processed_data['diagnosis'].map({'M': 1, 'B': 0})

    return processed_data

def normalize_input(user_input):
    data = load_data()
    features = data.drop(['diagnosis'], axis = 1) # removing the prediction from the data set
    normalized_value = {}

    for key, value in user_input.items():
        max = features[key].max()
        min = features[key].min()
        normalized_value[key] = (value - min) / (max - min)

    return normalized_value
