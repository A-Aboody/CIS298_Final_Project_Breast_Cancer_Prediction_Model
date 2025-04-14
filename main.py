import pandas as pd

def load_data():
    raw_data = pd.read_csv("Data/data.csv")

    processed_data = raw_data.drop(['Unnamed: 32', 'id'], axis=1)

    processed_data['diagnosis'] = processed_data['diagnosis'].map({'M': 1, 'B': 0})

    return processed_data

data = load_data()
print(data)