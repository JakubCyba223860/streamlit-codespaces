import time
import streamlit as st
import numpy as np
import joblib


st.title("EDA")

placeholder = st.empty()

progress_text = "Loading model and application data, please wait..."
my_bar = placeholder.progress(0, text=progress_text)
model = None
for percent_complete in range(100):
	if model == None:
		model = joblib.load("FinalDeliverable/app_data/artifacts/model_main.pkl")
	time.sleep(0.03)
	my_bar.progress(percent_complete + 1, text=progress_text)
	
time.sleep(0.5)
placeholder.empty()



st.subheader("Data Summary")
st.markdown("""<div style='border: 1px solid #ccc; padding: 10px;'>Provide a summary of the dataset used for analysis.</div>""", unsafe_allow_html=True)

st.subheader("Exploratory Analysis")
st.markdown("""<div style='border: 1px solid #ccc; padding: 10px;'>Perform exploratory analysis and showcase key insights.</div>""", unsafe_allow_html=True)


if model == None:
	st.warning("Couldn't find a valid model to load. Check artifacts/model_main.pkl!", icon="⚠️")
elif model != "validmodl":
	st.warning("Found model, but it has incorrect indexes. Check artifacts/model_main.pkl!", icon="⚠️")
else:
	# Define the input variables
	variable1 = st.sidebar.number_input("Property", value=0.0)
	variable2 = st.sidebar.number_input("Variable 2", value=0.0)
	variable3 = st.sidebar.number_input("Variable 3", value=0.0)
	variable4 = st.sidebar.number_input("Variable 4", value=0.0)
	variable5 = st.sidebar.number_input("Variable 5", value=0.0)
	variable6 = st.sidebar.number_input("Variable 6", value=0.0)
	with st.spinner('Loading EDA'):
		time.sleep(0.5)
		st.success('Done!')
	# Predict the value using the input variables
	input_data = np.array([[variable1, variable2, variable3, variable4, variable5, variable6]])
	prediction = model.predict(input_data)
	
	## Display the prediction in a Streamlit card
	st.write("Prediction:")
	st.info(prediction)
