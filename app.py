import streamlit as st
import joblib

# Load your saved model
model = joblib.load('Malpred.joblib')

# Label mapping
label_mapping = {1: "Malaria", 0: "No Malaria"}

# Define the prediction function
def predict_malaria(input_data):
    prediction = model.predict([input_data])[0]  # Predict the class (0 or 1)
    return label_mapping[prediction]

# Streamlit App
st.title("Malaria Prediction App")
st.write("This app predicts the likelihood of malaria based on symptoms. Please provide the following information:")

# Input fields for features
fever = st.number_input("Fever (1 if present, 0 if absent)", min_value=0, max_value=1, step=1)
cold = st.number_input("Cold (1 if present, 0 if absent)", min_value=0, max_value=1, step=1)
rigor = st.number_input("Rigor (1 if present, 0 if absent)", min_value=0, max_value=1, step=1)
fatigue = st.number_input("Fatigue (1 if present, 0 if absent)", min_value=0, max_value=1, step=1)
headache = st.number_input("Headache (1 if present, 0 if absent)", min_value=0, max_value=1, step=1)
bitter_tongue = st.number_input("Bitter Tongue (1 if present, 0 if absent)", min_value=0, max_value=1, step=1)
vomiting = st.number_input("Vomiting (1 if present, 0 if absent)", min_value=0, max_value=1, step=1)
diarrhea = st.number_input("Diarrhea (1 if present, 0 if absent)", min_value=0, max_value=1, step=1)

# Prediction button
if st.button("Predict"):
    # Prepare input data
    input_data = [fever, cold, rigor, fatigue, headache, bitter_tongue, vomiting, diarrhea]
    result = predict_malaria(input_data)
    st.success(f"The result is: {result}")
