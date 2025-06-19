from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_documents():
    loader = DirectoryLoader('data', glob="**/*.pdf")
    docs = loader.load()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    store = FAISS.from_documents(docs, embeddings)
    store.save_local("index")
    print("Ingestion completed and index saved.")

if __name__ == "__main__":
    ingest_documents()
