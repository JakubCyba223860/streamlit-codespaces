import streamlit as st
import numpy as np
import joblib

st.title("Why Extra Gradient Boost?")
st.write("We chose the model by implementing a rigorous evaluation process that involved the creation of multiple machine learning models by various team members. After comparing and cross-referencing the results, carefully analyzing graphs and plots, the Extreme Gradient Boost (XGBoost) algorithm consistently demonstrated superior performance, showcasing its ability to handle complex datasets and deliver exceptional results.")

st.title("Machine learning life cycle")
st.write("Throughout the ML lifecycle, we meticulously followed each step to develop and deploy the selected machine learning model. We collected and preprocessed data, trained and fine-tuned the model using Extreme Gradient Boost, and evaluated its performance through metrics such as Mean Absolute Error and Mean Square Error. Additionally, we incorporated a collaborative approach by encouraging everyone to create their own models. This involved team members developing their own machine learning models and comparing the results. By cross-referencing the performance metrics, analyzing graphs and plots, and conducting rigorous evaluations, we determined that the Extreme Gradient Boost algorithm consistently delivered the best results, solidifying its selection as the chosen model. This iterative process allowed us to make data-driven decisions and ensure the model's effectiveness in meeting our objectives.")

st.title("Prediction using our model")

# Load the pre-trained model
model = joblib.load("model_kornelia_dashboard.sav")

# Define the input variables
variable1 = st.number_input("Population density (km2)", value=1, step=1)
variable2 = st.number_input("Slow Response Time Penalty", value=1.0)
variable3_input = st.text_input("Trade and Catering %", value="1.0")
variable3 = float(variable3_input.rstrip('%')) if '%' in variable3_input else 1.0
variable4 = st.number_input("Response Time Score", value=1.0)
variable5_input = st.text_input("Percent uninhabited (%)", value="1.0")
variable5 = float(variable5_input.rstrip('%')) if '%' in variable5_input else 1.0



# Create an input data array
input_data = np.array([[variable1, variable2, variable3, variable4, variable5]])

# Predict the value using the input variables
prediction = model.predict(input_data)

# Display the prediction
st.write("Burglaries1k:")
st.info(prediction)
