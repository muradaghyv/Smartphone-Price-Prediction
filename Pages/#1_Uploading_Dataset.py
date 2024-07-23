import streamlit as st
import pandas as pd

st.title("File uploading")

if "data" in st.session_state:
    df = st.session_state["data"]
    st.write(df.head())
else:
    uploaded_file = st.file_uploader("", type=["csv","xlsx"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["data"] = df
        st.write(df.head())