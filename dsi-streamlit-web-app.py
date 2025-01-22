# Import libraries:
import streamlit as st
import pandas as pd
import joblib

# Load model pipeline object:
model = joblib.load("model.joblib")

# Add title and instructions:
st.title("Purchase Prediction Model")
st.subheader("Enter customer information and submit for likelihood to purchase")


# Add age input form:
age = st.number_input(
    label = "01. Enter the customer's age:",
    min_value = 18,
    max_value = 120,
    value = 35)

# Add gender input form:
gender = st.radio(
    label = "02. Enter the customer's gender:",
    options = ['M','F'])
    
# Add credit score input form:
credit_score = st.number_input(
    label = "03. Enter the customer's credit score:",
    min_value = 0,
    max_value = 1000,
    value = 500)

# Add submit button for submitting inputs to model:
if st.button("Submit For Prediction"):
    
    # Store data into a dataframe for prediction:
    new_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "credit_score": [credit_score]
        })
    
    # Apply model pipeline to the inputted data and extract probability prediction:
    pred_proba = model.predict_proba(new_data)[0][1]
    
    # Display prediction on web page:
    st.subheader(f"Based on these customer attributes, our model predicts a purchase probability of {pred_proba:.0%}")