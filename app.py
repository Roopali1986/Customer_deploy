import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model safely
filename = 'Model.sav'

try:
    with open(filename, 'rb') as file:
        load_model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit app title
st.title("Customer Churn Prediction")

# Input features
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure", min_value=0, max_value=100, value=1)
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=29.85)
TotalCharges = st.number_input("Total Charges", min_value=0.0, value=556.85)

# Create a DataFrame from user inputs
data = [[Dependents, tenure, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, Contract, PaperlessBilling, MonthlyCharges, TotalCharges]]
df = pd.DataFrame(data, columns=['Dependents', 'tenure', 'OnlineSecurity',
                                 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'Contract',
                                 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges'])

# Preprocessing: Encode categorical variables
encoder = LabelEncoder()
categorical_features = ['Dependents', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'Contract', 'PaperlessBilling']

for feature in categorical_features:
    df[feature] = encoder.fit_transform(df[feature])

# Make prediction
if st.button("Predict"):
    try:
        single = load_model.predict(df)
        probability = load_model.predict_proba(df)[:, 1]

        if single == 1:
            st.error("This Customer is likely to Churn!")
            st.write(f"Confidence level: {np.round(probability*100, 2)}%")
        else:
            st.success("This Customer is likely to Continue!")
            st.write(f"Confidence level: {np.round(probability*100, 2)}%")

    except Exception as e:
        st.error(f"Error making prediction: {e}")

# Ensure proper script execution
if __name__ == '__main__':
    st.write("App is running...")
 

