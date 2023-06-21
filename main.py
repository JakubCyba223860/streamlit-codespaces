import streamlit as st

st.set_page_config(page_title="Burglaries Prediction", page_icon=None, layout="centered", initial_sidebar_state="collapsed", menu_items=None)

st.markdown("# Team Eight: Burglaries Prediction")


def link_button(button_text: str, link: str):
   
    button_id = "DLbutton"

    custom_css = f""" 
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;

            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a target="_self" id="{button_id}" href="{link}">{button_text}</a><br></br>'

    return dl_link


st.markdown(link_button("Ready? Begin.", "starting"), unsafe_allow_html=True)
