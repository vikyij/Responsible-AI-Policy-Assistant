import streamlit as st
from src.extract_text import extract_text
from src.chunker import chunk_pages

st.set_page_config(page_title="Responsible AI Policy Assistant")

st.title("Responsible AI Policy Assistant") 

st.write("Upload AI policy documents and ask questions about them")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing document ..."):
            pages = extract_text(uploaded_file)
            chunk_pages(pages)
