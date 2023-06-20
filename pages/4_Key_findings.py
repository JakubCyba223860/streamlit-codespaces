import streamlit as st
import pandas as pd
import altair as alt
import time


st.title("Key Findings")

st.subheader("Summary")


insights_html = """
<ul>
  <li>EDA reveals a strong correlation between the presence of Points of Interest(POI) and the occurrence of violent crimes in the area.</li>
  <li>The percentage of elderly residents in a neighborhood is positively associated with the frequency of burglaries, indicating that older populations may be more vulnerable to such crimes.</li>
</ul>
"""

st.markdown("<div style='border: 1px solid #ccc; padding: 10px;'>{}</div>".format(insights_html), unsafe_allow_html=True)
