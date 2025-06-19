import os
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

def ingest(data_dir="data", index_dir="embeddings"):
    docs = []
    for f in os.listdir(data_dir):
        if f.lower().endswith(".pdf"):
            loader = UnstructuredPDFLoader(os.path.join(data_dir, f))
            docs.extend(loader.load())

    if not docs:
        print("⚠️ No documents found in the data folder. Please add PDFs.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(index_dir)
    print("✅ Ingestion complete.")

if __name__ == "__main__":
    ingest()
