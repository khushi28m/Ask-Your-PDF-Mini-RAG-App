# app.py

import streamlit as st
import os
import tempfile
from rag_pipeline import load_and_index, get_qa_chain

st.set_page_config(page_title="Mini LLM App", page_icon="🧠")
st.title("📄 Upload a PDF and Ask Questions (Mini Local LLM)")

# Upload PDF
uploaded_file = st.file_uploader("📤 Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir="docs") as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(uploaded_file.read())

    st.success("✅ File uploaded successfully. Indexing...")
    load_and_index(tmp_path)
    st.success("📚 Document indexed!")

    qa_chain = get_qa_chain()

    user_query = st.text_input("❓ Ask a question from your uploaded PDF:")

    if user_query:
        with st.spinner("Thinking..."):
            result = qa_chain.run(user_query)
            st.markdown(f"🧠 **Answer:** {result}")
else:
    st.info("👆 Upload a PDF file to begin.")
