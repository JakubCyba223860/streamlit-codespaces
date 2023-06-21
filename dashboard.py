import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import altair as alt


def render_introduction_page():
    """Renders the introduction page

    Arguments:

    Keyword Arguments:

    Returns:
        None
    """    

    st.title("Introduction")
    st.write("This is the content for the introduction page.")

    st.subheader("Background")
    st.markdown("""<div style='border: 1px solid #ccc; padding: 10px;'>Provide background information and context here.</div>""", unsafe_allow_html=True)

    st.subheader("Objective")
    st.markdown("""<div style='border: 1px solid #ccc; padding: 10px;'>Explain the objective or purpose of the project.</div>""", unsafe_allow_html=True)
    

def render_key_findings_page():
    """Renders the key findings page

    Arguments:

    Keyword Arguments:

    Returns:
        None
    """    
    st.title("Key Findings")
    st.write("This is the content for the key findings page.")

    st.subheader("Summary")
    st.markdown("""<div style='border: 1px solid #ccc; padding: 10px;'>
                        Here is a summary of the key findings:
                        - Finding 1: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
                        - Finding 2: Fusce et dui et urna volutpat tincidunt eu at velit.
                        - Finding 3: Phasellus non lobortis leo, vitae tincidunt augue.
                    </div>""", unsafe_allow_html=True)

    st.subheader("Visualizations")
    st.write("Here are some visualizations to support the key findings:")

    # Load data from CSV
    data = pd.read_csv("label_crime_allmetrics.csv")

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

    # Sort the grouped data in descending order based on the selected crime type
    grouped_data = grouped_data.sort_values(by=crime_type_filter, ascending=False)

    # Create an interactive bar chart using Altair
    chart = alt.Chart(grouped_data).mark_bar().encode(
        x="NeighbourhoodCode:N",
        y=crime_type_filter,
        tooltip=["NeighbourhoodCode", alt.Tooltip(crime_type_filter, format=".0f")]
    ).interactive()

    # Display the chart
    st.subheader("Bar Chart")
    st.altair_chart(chart, use_container_width=True)

    # Navigation buttons
    col1, col2, col3 = st.columns(3)

    # Button to go to the previous page
    if col1.button("Previous Page"):
        # Code to switch to the previous page goes here
        pass

    # Button to go to the next page
    if col3.button("Next Page"):
        # Code to switch to the next page goes here
        pass



def render_eda_page():
    """Renders the EDA page

    Arguments:

    Keyword Arguments:

    Returns:
        None
    """    
    st.title("EDA")
    st.write("This is the content for the EDA page.")

    st.subheader("Data Summary")
    st.markdown("""<div style='border: 1px solid #ccc; padding: 10px;'>Provide a summary of the dataset used for analysis.</div>""", unsafe_allow_html=True)

    st.subheader("Exploratory Analysis")
    st.markdown("""<div style='border: 1px solid #ccc; padding: 10px;'>Perform exploratory analysis and showcase key insights.</div>""", unsafe_allow_html=True)

    st.subheader("Sliders and Cards")
    data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5]

    # Create sliders
    slider1 = st.slider("Slider 1", min_value=0, max_value=10, value=5)
    slider2 = st.slider("Slider 2", min_value=0, max_value=10, value=5)
    slider3 = st.slider("Slider 3", min_value=0, max_value=10, value=5)
    slider4 = st.slider("Slider 4", min_value=0, max_value=10, value=5)
    slider5 = st.slider("Slider 5", min_value=0, max_value=10, value=5)
    slider6 = st.slider("Slider 6", min_value=0, max_value=10, value=5)
    slider7 = st.slider("Slider 7", min_value=0, max_value=10, value=5)
    slider8 = st.slider("Slider 8", min_value=0, max_value=10, value=5)

    # Update card values based on sliders
    card1_value = slider1 + slider2
    card2_value = slider3 - slider4
    card3_value = slider5 * slider6

    # Create cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div style='border: 1px solid #ccc; padding: 10px;'>Card 1 Value: {card1_value}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div style='border: 1px solid #ccc; padding: 10px;'>Card 2 Value: {card2_value}</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div style='border: 1px solid #ccc; padding: 10px;'>Card 3 Value: {card3_value}</div>", unsafe_allow_html=True)

    # Add predict button
    if st.button("Predict"):
        # Perform prediction or any desired action
        prediction = card1_value + card2_value + card3_value
        st.success(f"Prediction: {prediction}")

def render_ethics_page():
    """Renders the ethics page

    Arguments:

    Keyword Arguments:

    Returns:
        None
    """    
    st.title("Ethics")
    st.write("This is the content for the ethics page.")

    st.subheader("Ethical Considerations")
    st.markdown("""<div style='border: 1px solid #ccc; padding: 10px;'>Discuss the ethical considerations related to the project.</div>""", unsafe_allow_html=True)

    st.subheader("Data Privacy")
    st.markdown("""<div style='border: 1px solid #ccc; padding: 10px;'>Explain how data privacy was addressed in the project.</div>""", unsafe_allow_html=True)

def render_next_steps_page():
    """Renders the next steps page

    Arguments:

    Keyword Arguments:

    Returns:
        None
    """    
    st.title("Next Steps")
    st.write("This is the content for the next steps page.")

    st.subheader("Future Work")
    st.markdown("""<div style='border: 1px solid #ccc; padding: 10px;'>Outline potential future work and improvements.</div>""", unsafe_allow_html=True)

    st.subheader("Actionable Insights")
    st.markdown("""<div style='border: 1px solid #ccc; padding: 10px;'>Present actionable insights based on the project.</div>""", unsafe_allow_html=True)

def render_sources_page():
    """Renders the sources page

    Arguments:

    Keyword Arguments:

    Returns:
        None
    """    
    st.title("Sources")
    st.write("This is the content for the sources page.")

    st.subheader("Data Sources")
    st.markdown("""<div style='border: 1px solid #ccc; padding: 10px;'>List the sources of the data used in the project.</div>""", unsafe_allow_html=True)

    st.subheader("References")
    st.markdown("""<div style='border: 1px solid #ccc; padding: 10px;'>Provide references to relevant literature or resources.</div>""", unsafe_allow_html=True)

# Create a dictionary mapping page names to corresponding rendering functions
pages = {
    "Introduction": render_introduction_page,
    "Key Findings": render_key_findings_page,
    "EDA": render_eda_page,
    "Ethics": render_ethics_page,
    "Next Steps": render_next_steps_page,
    "Sources": render_sources_page
}

# Render the selected page based on user input
selected_page = st.sidebar.selectbox("Select a page", list(pages.keys()))
render_page = pages[selected_page]
render_page()
