import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ----------------------------
# Load model and scaler
# ----------------------------
model = pickle.load(open('parkinson_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Load dataset for feature names
data = pd.read_csv('parkinson.csv')
features = list(data.drop(columns=['name', 'status']).columns)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="🧬 Parkinson's Disease Detection", layout="centered")

st.title("🧠 Parkinson's Disease Detection")
st.write("This app uses an SVM classifier to detect Parkinson’s disease from biomedical voice measurements.")

st.sidebar.header("🧾 Input Patient Data")

# ----------------------------
# Sample Test Data
# ----------------------------
sample_data = {
    'MDVP:Fo(Hz)': 230.12345,
    'MDVP:Fhi(Hz)': 240.56789,
    'MDVP:Flo(Hz)': 220.98765,
    'MDVP:Jitter(%)': 0.00987,
    'MDVP:Jitter(Abs)': 0.00012,
    'MDVP:RAP': 0.00456,
    'MDVP:PPQ': 0.00512,
    'Jitter:DDP': 0.01234,
    'MDVP:Shimmer': 0.04567,
    'MDVP:Shimmer(dB)': 0.21000,
    'Shimmer:APQ3': 0.02012,
    'Shimmer:APQ5': 0.02567,
    'MDVP:APQ': 0.03045,
    'Shimmer:DDA': 0.05023,
    'NHR': 0.01234,
    'HNR': 18.76543,
    'RPDE': 0.567890,
    'DFA': 0.876543,
    'spread1': -5.432100,
    'spread2': 0.256789,
    'D2': 2.345678,
    'PPE': 0.123456
}


# sample_data_healthy = {
#     'MDVP:Fo(Hz)': 190.23456,
#     'MDVP:Fhi(Hz)': 200.34567,
#     'MDVP:Flo(Hz)': 185.12345,
#     'MDVP:Jitter(%)': 0.00123,
#     'MDVP:Jitter(Abs)': 0.00001,
#     'MDVP:RAP': 0.00145,
#     'MDVP:PPQ': 0.00150,
#     'Jitter:DDP': 0.00321,
#     'MDVP:Shimmer': 0.00890,
#     'MDVP:Shimmer(dB)': 0.08500,
#     'Shimmer:APQ3': 0.00450,
#     'Shimmer:APQ5': 0.00560,
#     'MDVP:APQ': 0.00670,
#     'Shimmer:DDA': 0.01450,
#     'NHR': 0.00310,
#     'HNR': 28.12345,
#     'RPDE': 0.412345,
#     'DFA': 0.723456,
#     'spread1': -7.876543,
#     'spread2': 0.167890,
#     'D2': 1.654321,
#     'PPE': 0.080123
# }


# ----------------------------
# User Input Section
# ----------------------------
inputs = []
use_sample = st.sidebar.checkbox("🧪 Use Sample Data", value=True)

for feature in features:
    mean_val = float(data[feature].mean())
    default_val = float(sample_data.get(feature, mean_val)) if use_sample else mean_val
    # ✅ Remove min/max to prevent Streamlit errors
    value = st.sidebar.number_input(f"{feature}", value=default_val, step=0.000001, format="%.6f")
    inputs.append(value)

# Convert to DataFrame for scaler
input_df = pd.DataFrame([inputs], columns=features)
scaled_input = scaler.transform(input_df)

# ----------------------------
# Prediction Button
# ----------------------------
if st.button("Predict"):
    prediction = model.predict(scaled_input)
    if prediction[0] == 0:
        st.success("✅ The person **does not have Parkinson’s Disease**.")
    else:
        st.error("⚠️ The person **has Parkinson’s Disease**.")

# ----------------------------
# Footer
# ----------------------------
st.caption("Developed with ❤️ using Streamlit and Scikit-learn")
