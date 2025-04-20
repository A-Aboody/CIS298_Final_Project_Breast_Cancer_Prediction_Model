import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def load_data():
    raw_data = pd.read_csv("data/data.csv")

    processed_data = raw_data.drop(['Unnamed: 32', 'id'], axis=1)

    processed_data['diagnosis'] = processed_data['diagnosis'].map({'M': 1, 'B': 0})

    return processed_data

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

def normalize_input(user_input):
    dataset = load_data()

    features = dataset.drop(['diagnosis'], axis=1)

    normalized_values = {}

    for key, value in user_input.items():
        max_val = features[key].max()
        min_val = features[key].min()
        normalized_values[key] = (value - min_val) / (max_val - min_val)

    return normalized_values

def create_radar_chart(normalized_input):
    normalized_input = normalize_input(normalized_input)

    measurement_categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                              'Smoothness', 'Compactness', 
                              'Concavity', 'Concave Points',
                              'Symmetry', 'Fractal Dimension']

    chart = go.Figure()

    chart.add_trace(go.Scatterpolar(
        r=[
            normalized_input['radius_mean'], normalized_input['texture_mean'], normalized_input['perimeter_mean'],
            normalized_input['area_mean'], normalized_input['smoothness_mean'], normalized_input['compactness_mean'],
            normalized_input['concavity_mean'], normalized_input['concave points_mean'], normalized_input['symmetry_mean'],
            normalized_input['fractal_dimension_mean']
        ],
        theta=measurement_categories,
        fill='toself',
        name='Mean Value'
    ))

    chart.add_trace(go.Scatterpolar(
        r=[
            normalized_input['radius_se'], normalized_input['texture_se'], normalized_input['perimeter_se'],
            normalized_input['area_se'], normalized_input['smoothness_se'], normalized_input['compactness_se'],
            normalized_input['concavity_se'], normalized_input['concave points_se'], normalized_input['symmetry_se'],
            normalized_input['fractal_dimension_se']
        ],
        theta=measurement_categories,
        fill='toself',
        name='Standard Error'
    ))

    chart.add_trace(go.Scatterpolar(
        r=[
            normalized_input['radius_worst'], normalized_input['texture_worst'], normalized_input['perimeter_worst'],
            normalized_input['area_worst'], normalized_input['smoothness_worst'], normalized_input['compactness_worst'],
            normalized_input['concavity_worst'], normalized_input['concave points_worst'], normalized_input['symmetry_worst'],
            normalized_input['fractal_dimension_worst']
        ],
        theta=measurement_categories,
        fill='toself',
        name='Worst Value'
    ))

    chart.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    return chart

def make_predictions(user_input):
    trained_model = pickle.load(open("model/model.pkl", "rb"))
    trained_scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_features = np.array(list(user_input.values())).reshape(1, -1)

    scaled_features = trained_scaler.transform(input_features)

    prediction_result = trained_model.predict(scaled_features)

    st.subheader("Prediction Result")
    st.write("The cell cluster is predicted to be:")

    if prediction_result[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malignant'>Malignant</span>", unsafe_allow_html=True)

    st.write("Probability of being benign: ", trained_model.predict_proba(scaled_features)[0][0])
    st.write("Probability of being malignant: ", trained_model.predict_proba(scaled_features)[0][1])

def main():
    st.set_page_config(
        page_title="Breast Cancer Prediction Model",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with open("assets/style.css") as css_file:
        st.markdown("<style>{}</style>".format(css_file.read()), unsafe_allow_html=True)

    user_input = configure_sidebar()

    with st.container():
        st.title("Breast Cancer Prediction")

    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = create_radar_chart(user_input)
        st.plotly_chart(radar_chart)

    with col2:
        make_predictions(user_input)

if __name__ == '__main__':
    main()
