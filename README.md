This project is a mini Retrieval-Augmented Generation (RAG) system built using Python, Streamlit, LangChain, and Hugging Face models. It allows users to upload a PDF, automatically indexes its contents, and then asks questions to retrieve relevant answers using embeddings and a lightweight LLM.

Project Structure:

- app.py : Streamlit app for PDF upload and question-answering

- rag_pipeline.py : Handles document loading, chunking, embeddings, FAISS index, and QA chain setup

Features:

- Upload a PDF and interact with its contents

- Splits documents into chunks for efficient retrieval

- Embeds text using HuggingFace MiniLM model

- Stores embeddings in FAISS vector database

- Uses a lightweight GPT-2 model for answer generation

- Streamlit interface for easy user interaction

Technologies Used:

- Programming Language: Python

- Frameworks & Libraries: Streamlit, LangChain, HuggingFace Transformers, FAISS, PyMuPDF

- Models: Sentence-Transformers MiniLM for embeddings, DistilGPT-2 for text generation

Getting Started:

- Prerequisites: Install the required libraries:
pip install streamlit langchain transformers sentence-transformers faiss-cpu pymupdf

- Run the Streamlit App:
streamlit run app.py

This will open the interactive web app in your browser.

Preview

- Upload a PDF document.

- Ask questions in plain text.

- Get AI-generated answers based on document context.

Future Improvements:

- Support for multiple document uploads

- Integration with larger LLMs (e.g., LLaMA, Falcon)
