# Breast Cancer Prediction Dashboard

A Streamlit web application that predicts breast cancer diagnosis (benign/malignant) based on cell nucleus characteristics.

![image](https://github.com/user-attachments/assets/ec9420db-13aa-44ac-a2a3-f5fda1ab46e2)


## Key Contributions

**Developer**: Abdula Ameen:

### Model Infrastructure
- Implemented Logistic Regression model with 95%+ accuracy
- Designed data preprocessing pipeline with StandardScaler
- Automated model training and serialization (pickle files)

### Prediction System
- Developed real-time prediction interface
- Added confidence percentage metrics for predictions
- Implemented input validation and error handling

/*
Michael and Safar add your contributions before technical stack
*/

## Technical Stack
- **Backend**: Python, Scikit-learn, Pandas, Pickle
- **Frontend**: Streamlit, Plotly
- **Data**: Wisconsin Breast Cancer Dataset

## How to Run
pip install -r requirements.txt <br>
python -m streamlit run App/main.py