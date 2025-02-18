import streamlit as st
import pandas as pd
import altair as alt
import time
from PIL import Image

st.title("EDA")

__application_path_prefix = "pages/"

placeholder = st.empty()
with st.spinner('Loading findings/metrics file...'):
    # Load data from CSV
    data = pd.read_csv(__application_path_prefix + "label_crime_allmetrics.csv")
    if data is not None:
        placeholder.success('Done!')
    else:
        placeholder.warning('No data found!')
        time.sleep(1)
    placeholder.empty()

# Filter data by year and month
year_filter = st.selectbox("Filter by Year", data["Year"].unique())
month_filter = st.selectbox("Filter by Month", data["Month"].unique())
filtered_data = data[(data["Year"] == year_filter) & (data["Month"] == month_filter)]

# Get the list of available crime types
crime_types = [column for column in filtered_data.columns if column not in ["NeighbourhoodCode", "Year", "Month"]]

# Choose the crime type as a filter
crime_type_filter = st.selectbox("Filter by Crime Type", crime_types)

# Group data by neighbourhood code and sum the selected crime type
grouped_data = filtered_data.groupby("NeighbourhoodCode")[crime_type_filter].sum().reset_index()

# Convert the 'NeighbourhoodCode' column to string type
grouped_data["NeighbourhoodCode"] = grouped_data["NeighbourhoodCode"].astype(str)

# Sort the grouped data in descending order based on the selected crime type
grouped_data = grouped_data.sort_values(by=crime_type_filter, ascending=False)

# Create an interactive bar chart using Altair
chart = alt.Chart(grouped_data).mark_bar().encode(
    x="NeighbourhoodCode:N",
    y=alt.Y(crime_type_filter, axis=alt.Axis(title="Count")),
    tooltip=["NeighbourhoodCode", alt.Tooltip(crime_type_filter, format=".0f")]
).interactive()

# Display the chart
st.subheader("Bar Chart")
st.altair_chart(chart, use_container_width=True)

# Navigation buttons
col1, col2, col3 = st.columns(3)

image = Image.open('pages/Corrmx.png')
st.image(image, caption='Correlation Matrix', use_column_width=True)
