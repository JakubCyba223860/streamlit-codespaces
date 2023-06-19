import streamlit as st

# Logo
# st.image("logo.png", use_column_width=True)

# Team Name
st.title("Team Eight")
st.subheader("Our team:")

# Define member information
members = [
    {
        "name": "Flizik, Kornelia",
        "role": "Data Scientist, Analytics Translator",
        "github": "https://github.com/KorneliaFlizik223643",
        "id": "223643",
        # "photo": "member1_photo.png"
    },
    {
        "name": "Cyba, Jakub",
        "role": "Data Engineer, App Developer",
        "github": "https://github.com/JakubCyba223860",
        "id": "223860",
        # "photo": "member4_photo.png"
        
    },
    {
        "name": "Rákosi, Mark",
        "role": "Data Engineer, App Developer",
        "github": "https://github.com/markrakosi225087",
        "id": "225087",
        # "photo": "member3_photo.png"
    },
    {
        "name": "Graziadei, Benjamin",
        "role": "Data Scientist, Data Engineer",
        "github": "https://github.com/BenjaminGraziadei223946",
        "id": "223946",
        # "photo": "member2_photo.png"
    },
    {
        "name": "Oomes, Shan",
        "role": "Data Scientist, Data Engineer",
        "github": "https://github.com/ShanToa",
        "id": "220231",
        # "photo": "member5_photo.png"
    }
]

# Display member information in a table
st.markdown('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns([1, 2, 4, 2])
with col2:
    st.markdown("##### Name, ID")
with col3:
    st.markdown("##### Role")
with col4:
    st.markdown("##### Github")
        
for member in members:
    # member_photo = member["photo"]
    member_name = member["name"]
    member_role = member["role"]
    member_github = member["github"]
    member_number = member["id"]

    col1, col2, col3, col4 = st.columns([1, 2, 4, 3])
    # with col1:
        # st.image(member_photo, width=80)
    with col2:
        st.write(f"**{member_name}**")
    with col3:
        st.write(f"{member_role}")
    with col4:
        st.write(f"{member_github}")
    with col2:
        st.write(f"_({member_number})_")

    st.markdown("---")


st.balloons()