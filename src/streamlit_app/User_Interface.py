"""User interface using the Streamlit package"""

# Enter the following into the command line, then refresh the browser
#   or save an updated file to see updates:
# pip install streamlit
# streamlit run User_Interface.py

######################################################################
import streamlit as st

st.set_page_config(
    page_title="Hello Rudolf",
    page_icon="👋",
    layout="wide",
)

st.title("Polarization Light Field Microscopy Simulations")

st.markdown("*There are tabs on the left. Click around to explore.*")
