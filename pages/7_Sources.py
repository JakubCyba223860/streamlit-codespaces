import streamlit as st


st.title("Sources")
st.write("This is the content for the sources page.")

data_sources = [
    "Breda. (n.d.). Breda. (n.d.). [Link](https://breda.incijfers.nl/Dashboard)",
    "Netherlands, S. (2023, May 17). CBS - Statistics Netherlands. Statistics Netherlands. [Link](https://www.cbs.nl/en-gb)",
    "Home - Eurostat. (n.d.). Eurostat. [Link](https://ec.europa.eu/eurostat)",
    "CBS Statline. (n.d.). https://data.politie.nl/#/Politie/nl/"
]

st.subheader("Data Sources")
for source in data_sources:
    st.markdown(f"<div style='border: 1px solid #ccc; padding: 10px;'>{source}</div>", unsafe_allow_html=True)
