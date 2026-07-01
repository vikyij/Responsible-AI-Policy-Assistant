import streamlit as st
from src.extract_text import extract_text
from src.chunker import chunk_pages
from src.embeddings import create_embedding
from src.vector_store import store_chunks, reset_store
from src.rag_pipeline import answer_question, generate_responsible_ai_checklist, perform_gap_analysis

if "document_indexed" not in st.session_state:
    st.session_state.document_indexed = False

if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""

if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None


st.set_page_config(page_title="Responsible AI Policy Assistant")

st.title("Responsible AI Policy Assistant") 

st.write("Upload AI policy documents and ask questions about them")


uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    if st.session_state.current_file_name != uploaded_file.name:
        st.session_state.document_indexed = False
        st.session_state.selected_question = ""
        st.session_state.current_file_name = uploaded_file.name



if uploaded_file and not st.session_state.document_indexed:
    reset_store()
    with st.spinner("Processing document ..."):
            pages = extract_text(uploaded_file)
            chunks = chunk_pages(pages)

            for chunk in chunks:
                  chunk["embedding"] = create_embedding(chunk["text"])

            store_chunks(chunks, uploaded_file.name)

    st.session_state.document_indexed = True
    st.success("Document indexed successfully.")

if st.session_state.document_indexed:
    st.subheader("Suggested Questions")

    suggested_questions = [
        "What does this document say about fairness/bias?",
        "What risks are identified in this document?",
        "Does this document mention human oversight?",
        "What accountability mechanisms are described?",
        "What does this document say about transparency?",
        "What gaps exist in this policy?",
        "What does this document say about privacy?",
    ]

    for question_text in suggested_questions:
        if st.button(question_text):
            st.session_state.selected_question = question_text

    question = st.text_input(
        "Ask a question about the document",
        value=st.session_state.selected_question
    )
    
    if question:
      with st.spinner("Searching document and generating answer..."):
            answer, sources = answer_question(question)

      st.subheader("Answer")
      st.write(answer)

      st.subheader("Sources")
      for source in sources:
            st.markdown(f"**{source['document']} | Page {source['page']}**")
            st.write(source["text"][:500])
            st.caption(f"Similarity score: {source['score']:.4f}")

    st.subheader("Responsible AI Checklist")
    if st.button("Generate Responsible AI Checklist"):
        with st.spinner("Generating checklist..."):
            checklist_answer, checklist_sources = generate_responsible_ai_checklist()

        st.subheader("Checklist")
        st.write(checklist_answer)

        st.subheader("Checklist Sources")
        for source in checklist_sources:
            st.markdown(f"**{source['document']} | Page {source['page']}**")
            st.write(source["text"][:500])
            st.caption(f"Similarity score: {source['score']:.4f}")

            
    st.subheader("Responsible AI Gap Analysis")

    if st.button("Generate Gap Analysis"):
        with st.spinner("Generating gap analysis..."):
            gap_answer, gap_sources = perform_gap_analysis()

        st.subheader("Gap Analysis")
        st.write(gap_answer)

