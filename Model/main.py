import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

def train_model(cleaned_data): 
    features = cleaned_data.drop(['diagnosis'], axis=1)
    target = cleaned_data['diagnosis']

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    features_train, features_test, target_train, target_test = train_test_split(
        scaled_features, target, test_size=0.2, random_state=42
    )

    classifier = LogisticRegression()
    classifier.fit(features_train, target_train)

    predictions = classifier.predict(features_test)
    print('Model Accuracy: ', accuracy_score(target_test, predictions))
    print("Classification Report: \n", classification_report(target_test, predictions))

    return classifier, scaler

def load_and_clean_data():
    raw_data = pd.read_csv("data/data.csv")

    cleaned_data = raw_data.drop(['Unnamed: 32', 'id'], axis=1)

    cleaned_data['diagnosis'] = cleaned_data['diagnosis'].map({ 'M': 1, 'B': 0 })

    return cleaned_data

def main():
    processed_data = load_and_clean_data()

    trained_model, trained_scaler = train_model(processed_data)

    with open('model/model.pkl', 'wb') as model_file:
        pickle.dump(trained_model, model_file)

    with open('model/scaler.pkl', 'wb') as scaler_file:
        pickle.dump(trained_scaler, scaler_file)

if __name__ == '__main__':
    main()
