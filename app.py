import streamlit as st

st.set_page_config(page_title="Responsible AI Policy Assistant")

st.title("Responsible AI Policy Assistant") 

st.write("Upload AI policy documents and ask questions about them")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")