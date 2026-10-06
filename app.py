import streamlit as st
import pandas as pd
import joblib

model = joblib.load("diabetes_random_forest.pkl")

st.title("Diabetes Prediction")
st.write("Enter patient details")

data = {}

#for column in model.named_steps["preprocessor"].feature_names_in_:
 #   data[column] = st.text_input(column)
features = [
    "Fasting_Blood_Glucose",
    "Postprandial_Blood_Glucose",
    "HbA1c",
    "Random_Blood_Glucose",
    "BMI",
    "Waist_Circumference",
    "Blood_Pressure_Systolic",
    "Blood_Pressure_Diastolic",
    "LDL_Cholesterol",
    "HDL_Cholesterol",
    "Family_History_of_Diabetes",
    "Physical_Activity"
]

#for column in features:
 #   data[column] = st.text_input(column)
for column in features:
    if column in ["Family_History_of_Diabetes", "Physical_Activity"]:
        data[column] = st.text_input(column)
    else:
        data[column] = st.number_input(column, value=0.0)

if st.button("Predict"):
    input_data = pd.DataFrame([data])

    prediction = model.predict(input_data)

    if prediction[0] == "Positive":
        st.error("Diabetes Detected")
    else:
        st.success("No Diabetes Detected")

    