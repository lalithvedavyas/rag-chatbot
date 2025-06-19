import os
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def ingest_documents():
    # Load documents from the 'data' folder
    loader = DirectoryLoader('data', glob="**/*.pdf")
    docs = loader.load()

    # Initialize free embeddings from Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vector store from documents and embeddings
    store = FAISS.from_documents(docs, embeddings)

    # Save the index locally in 'index' folder
    store.save_local("index")
    print("Ingestion completed and index saved.")

if __name__ == "__main__":
    ingest_documents()
