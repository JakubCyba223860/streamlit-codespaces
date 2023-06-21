import streamlit as st
import numpy as np
import joblib
import time
st.title("Why Gradient Boost Regressor?")
st.write("We arrived at the decision to utilize the Gradient Boosting Regressor as our chosen model after an extensive evaluation process. Our team members created multiple machine learning models and conducted thorough comparisons and analyses of the results. After carefully examining graphs and plots, the Gradient Boosting Regressor algorithm consistently showcased remarkable performance, demonstrating its capability to handle complex datasets and deliver exceptional outcomes.")

st.title("Machine learning life cycle")
st.write("Throughout the machine learning life cycle, we meticulously followed each step to develop and deploy our selected model. We collected and preprocessed the data, trained and fine-tuned the model using Gradient Boosting Regressor, and evaluated its performance using metrics such as Mean Absolute Error and Mean Square Error. Additionally, we fostered a collaborative environment, encouraging each team member to create their own models. This involved developing individual machine learning models and comparing the results. By carefully examining performance metrics, analyzing graphs and plots, and conducting rigorous evaluations, we consistently found that the Gradient Boosting Regressor algorithm delivered superior results, solidifying its selection as our chosen model. This iterative process allowed us to make data-driven decisions and ensure the effectiveness of the model in meeting our objectives.")

st.title("Prediction using our model")

# Load the pre-trained model
model = joblib.load("model_kornelia_dashboard.sav")

# Define the input variables
variable1 = st.number_input("Population density (km2)", value=3109, step=1)
variable2 = st.number_input("Slow Response Time Penalty", value=1.0)
variable3_input = st.text_input("Trade and Catering %", value="0.141")
variable3 = float(variable3_input.rstrip('%')) if '%' in variable3_input else 0.141
variable4 = st.number_input("Response Time Score", value=0.0)
variable5_input = st.text_input("Percent uninhabited (%)", value="1.0")
variable5 = float(variable5_input.rstrip('%')) if '%' in variable5_input else 2



# Create an input data array
with st.spinner('Predicting values...'):
    input_data = np.array([[variable1, variable2, variable3, variable4, variable5]])
    time.sleep(0.3)
    # Predict the value using the input variables
    prediction = model.predict(input_data)
    time.sleep(0.3)
    # Display the prediction
    st.write("Burglaries1k:")
    st.info(prediction)
    st.success('Done!')


