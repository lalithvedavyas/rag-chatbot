from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def ingest_documents():
    # Load PDFs from 'data' folder recursively
    loader = DirectoryLoader('data', glob="**/*.pdf")
    docs = loader.load()

    # Optional: Split documents into smaller chunks for better embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(docs)

    # Initialize HuggingFace embeddings (all-MiniLM-L6-v2)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS index from docs and embeddings
    store = FAISS.from_documents(docs, embeddings)

    # Save index locally to 'faiss_index' folder (match chatbot loading path)
    store.save_local("faiss_index")

    print("Ingestion completed and FAISS index saved in 'faiss_index/' folder.")

if __name__ == "__main__":
    ingest_documents()
