import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def load_data():
    try:
        raw_data = pd.read_csv("data/data.csv")
    except FileNotFoundError:
        st.error("Error: data/data.csv not found. Please ensure the data file exists.")
        return None

    processed_data = raw_data.drop(columns=['Unnamed: 32', 'id'], errors='ignore')

    if 'diagnosis' in processed_data.columns:
        processed_data['diagnosis'] = processed_data['diagnosis'].map({'M': 1, 'B': 0})
    else:
        st.error("Error: 'diagnosis' column not found in the data.")
        return None

    expected_features = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
        'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    missing_cols = [col for col in expected_features if col not in processed_data.columns]
    if missing_cols:
        st.error(f"Error: The following expected feature columns are missing: {', '.join(missing_cols)}")
        return None

    return processed_data

def configure_sidebar(dataset):
    st.sidebar.header("Input Measurements")
    st.sidebar.caption("Adjust the values below")

    measurement_labels = {
        "Mean": [
            ("Radius", "radius_mean"), ("Texture", "texture_mean"), ("Perimeter", "perimeter_mean"),
            ("Area", "area_mean"), ("Smoothness", "smoothness_mean"), ("Compactness", "compactness_mean"),
            ("Concavity", "concavity_mean"), ("Concave points", "concave points_mean"),
            ("Symmetry", "symmetry_mean"), ("Fractal dimension", "fractal_dimension_mean")
        ],
        "Standard Error": [
            ("Radius", "radius_se"), ("Texture", "texture_se"), ("Perimeter", "perimeter_se"),
            ("Area", "area_se"), ("Smoothness", "smoothness_se"), ("Compactness", "compactness_se"),
            ("Concavity", "concavity_se"), ("Concave points", "concave points_se"),
            ("Symmetry", "symmetry_se"), ("Fractal dimension", "fractal_dimension_se")
        ],
        "Worst": [
            ("Radius", "radius_worst"), ("Texture", "texture_worst"), ("Perimeter", "perimeter_worst"),
            ("Area", "area_worst"), ("Smoothness", "smoothness_worst"), ("Compactness", "compactness_worst"),
            ("Concavity", "concavity_worst"), ("Concave points", "concave points_worst"),
            ("Symmetry", "symmetry_worst"), ("Fractal dimension", "fractal_dimension_worst")
        ]
    }

    user_input = {}

    for category, items in measurement_labels.items():
        with st.sidebar.expander(f"{category} Values", expanded=(category == "Mean")):
            for label, key in items:
                if key in dataset.columns:
                    user_input[key] = st.slider(
                        label=f"{label}",
                        min_value=float(0),
                        max_value=float(dataset[key].max()),
                        value=float(dataset[key].mean()),
                        key=key,
                        format="%.4f"
                    )
                else:
                    user_input[key] = 0

    return user_input

def normalize_input(user_input, dataset):
    features = dataset.drop(['diagnosis'], axis=1)
    normalized_values = {}
    for key, value in user_input.items():
        if key in features.columns:
            max_val = features[key].max()
            min_val = features[key].min()
            normalized_values[key] = 0 if max_val == min_val else (value - min_val) / (max_val - min_val)
        else:
             normalized_values[key] = 0
    return normalized_values

def create_radar_chart(user_input, dataset):
    normalized_input_vals = normalize_input(user_input, dataset)
    measurement_categories = ['Radius', 'Texture', 'Perimeter', 'Area',
                             'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                             'Symmetry', 'Fractal Dimension']
    color_sequence = ['#1f77b4', '#ff7f0e', '#2ca02c']
    fill_alpha = 0.4
    rgba_colors = [
        f'rgba(31, 119, 180, {fill_alpha})',
        f'rgba(255, 127, 14, {fill_alpha})',
        f'rgba(44, 160, 44, {fill_alpha})'
    ]

    chart = go.Figure()

    chart.add_trace(go.Scatterpolar(
        r=[normalized_input_vals.get(f'{cat.lower().replace(" ", "")}_mean', 0) for cat in measurement_categories],
        theta=measurement_categories,
        fill='toself',
        name='Mean Value',
        line=dict(color=color_sequence[0]),
        fillcolor=rgba_colors[0]
    ))

    chart.add_trace(go.Scatterpolar(
        r=[normalized_input_vals.get(f'{cat.lower().replace(" ", "")}_se', 0) for cat in measurement_categories],
        theta=measurement_categories,
        fill='toself',
        name='Standard Error',
        line=dict(color=color_sequence[1]),
        fillcolor=rgba_colors[1]
    ))

    chart.add_trace(go.Scatterpolar(
        r=[normalized_input_vals.get(f'{cat.lower().replace(" ", "")}_worst', 0) for cat in measurement_categories],
        theta=measurement_categories,
        fill='toself',
        name='Worst Value',
        line=dict(color=color_sequence[2]),
        fillcolor=rgba_colors[2]
    ))

    chart.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            angularaxis=dict(linecolor="#cccccc", gridcolor="#777777"),
            radialaxis=dict(visible=True, range=[0, 1], linecolor="#cccccc", gridcolor="#777777", tickfont=dict(color='#ffffff'))
        ),
        showlegend=True,
        legend=dict(font=dict(color="#ffffff")),
        title=dict(text="Normalized Cell Measurements", x=0.5, font=dict(color='#ffffff')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=80, b=40)
    )

    return chart

def display_predictions(user_input):
    try:
        trained_model = pickle.load(open("model/model.pkl", "rb"))
        trained_scaler = pickle.load(open("model/scaler.pkl", "rb"))
    except FileNotFoundError:
        st.error("Error: Model or scaler file not found.")
        return
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        return

    feature_keys_ordered = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
        'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    input_values = [user_input.get(key, 0) for key in feature_keys_ordered]
    input_features = np.array(input_values).reshape(1, -1)

    try:
        scaled_features = trained_scaler.transform(input_features)
        prediction_result = trained_model.predict(scaled_features)
        prediction_proba = trained_model.predict_proba(scaled_features)
    except Exception as e:
         st.error(f"Error during prediction: {e}")
         return

    st.header("Prediction Result")
    if prediction_result[0] == 0:
        st.markdown("<div class='diagnosis-card benign'><h3>Benign</h3></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='diagnosis-card malignant'><h3>Malignant</h3></div>", unsafe_allow_html=True)

    st.metric(label="Confidence (Benign)", value=f"{prediction_proba[0][0]:.1%}")
    st.metric(label="Confidence (Malignant)", value=f"{prediction_proba[0][1]:.1%}")

def main():
    st.set_page_config(
        page_title="Breast Cancer Prediction Model",
        page_icon="./BCPM.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    try:
        with open("assets/style.css") as css_file:
            st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown("""
            <style>
            body { background-color: #0E1117; color: #FAFAFA; }
            .stApp { background-color: #0E1117; }
            </style>
        """, unsafe_allow_html=True)

    dataset = load_data()
    if dataset is None:
        st.stop()

    user_input_values = configure_sidebar(dataset)

    st.title("Breast Cancer Prediction Dashboard")

    col1, col2 = st.columns([3, 1], gap="large")

    with col1:
        st.header("Cell Measurements Visualization")
        radar_chart = create_radar_chart(user_input_values, dataset)
        st.plotly_chart(radar_chart, use_container_width=True)

    with col2:
        display_predictions(user_input_values)

if __name__ == '__main__':
    main()
