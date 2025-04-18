import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

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

def configure_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")

    dataset = load_data()
    measurement_labels = [
        ("Radius (Mean)", "radius_mean"),
        ("Texture (Mean)", "texture_mean"),
        ("Perimeter (Mean)", "perimeter_mean"),
        ("Area (Mean)", "area_mean"),
        ("Smoothness (Mean)", "smoothness_mean"),
        ("Compactness (Mean)", "compactness_mean"),
        ("Concavity (Mean)", "concavity_mean"),
        ("Concave points (Mean)", "concave points_mean"),
        ("Symmetry (Mean)", "symmetry_mean"),
        ("Fractal dimension (Mean)", "fractal_dimension_mean"),
        ("Radius (Standard Error)", "radius_se"),
        ("Texture (Standard Error)", "texture_se"),
        ("Perimeter (Standard Error)", "perimeter_se"),
        ("Area (Standard Error)", "area_se"),
        ("Smoothness (Standard Error)", "smoothness_se"),
        ("Compactness (Standard Error)", "compactness_se"),
        ("Concavity (Standard Error)", "concavity_se"),
        ("Concave points (Standard Error)", "concave points_se"),
        ("Symmetry (Standard Error)", "symmetry_se"),
        ("Fractal dimension (Standard Error)", "fractal_dimension_se"),
        ("Radius (Worst)", "radius_worst"),
        ("Texture (Worst)", "texture_worst"),
        ("Perimeter (Worst)", "perimeter_worst"),
        ("Area (Worst)", "area_worst"),
        ("Smoothness (Worst)", "smoothness_worst"),
        ("Compactness (Worst)", "compactness_worst"),
        ("Concavity (Worst)", "concavity_worst"),
        ("Concave points (Worst)", "concave points_worst"),
        ("Symmetry (Worst)", "symmetry_worst"),
        ("Fractal dimension (Worst)", "fractal_dimension_worst"),
    ]

    user_input = {}
    for label, key in measurement_labels:
        user_input[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(dataset[key].max()),
            value=float(dataset[key].mean())
        )

    return user_input