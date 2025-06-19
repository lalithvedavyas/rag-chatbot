from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

def ingest_documents():
    # Path to your documents folder
    data_path = "data"

    # Collect all document file paths (example for PDFs)
    pdf_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".pdf")]

    docs = []
    for pdf in pdf_files:
        loader = UnstructuredPDFLoader(pdf)
        docs.extend(loader.load())

    # Initialize embeddings model
    embeddings = OpenAIEmbeddings()

    # Create FAISS vector store from documents
    store = FAISS.from_documents(docs, embeddings)

    # Save the FAISS index locally in embeddings folder
    store.save_local("embeddings")

if __name__ == "__main__":
    ingest_documents()
    print("Document ingestion and vector store creation completed.")
