# rag_pipeline.py

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from transformers import pipeline


def load_and_index(path):
    # Load and split PDF
    loader = PyMuPDFLoader(path)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Local CPU embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # Create FAISS index
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local("faiss_index")


def get_qa_chain():
    # Load FAISS index with same embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectordb = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Use a lightweight text generation model
    qa_pipe = pipeline("text-generation", model="distilgpt2", device=-1, max_new_tokens=150)
    llm = HuggingFacePipeline(pipeline=qa_pipe)

    # RAG pipeline
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
    return qa
