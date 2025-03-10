# prompt: add streamlit app code with it for the above code  in seprate file app.py in vs code add main add --main--- as this is py file

from ast import main
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model
filename = 'Model.sav'
load_model = pickle.load(open(filename, 'rb'))

# Streamlit app
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

# Preprocess the input data (same as in the original notebook)
encoder = LabelEncoder()
for feature in df.columns:
    if df[feature].dtype == 'object':
        df[feature] = encoder.fit_transform(df[feature])
        
# Make prediction
single = load_model.predict(df)
probability = load_model.predict_proba(df)[:, 1]


# Display prediction results
if st.button("Predict"):
  if single == 1:
      st.write("This Customer is likely to be Churned!")
      st.write(f"Confidence level is {np.round(probability*100, 2)}%")
  else:
      st.write("This Customer is likely to continue!")
      st.write(f"Confidence level is {np.round(probability*100, 2)}%")




if __name__ == '__main__':
   main()
