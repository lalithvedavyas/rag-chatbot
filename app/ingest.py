import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

def ingest_documents(data_dir="data", index_dir="embeddings"):
    docs = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if filename.lower().endswith(".pdf"):
            loader = UnstructuredPDFLoader(filepath)
        else:
            # Add other loaders for DOCX, TXT etc, or skip
            continue

        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_split = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    store = FAISS.from_documents(docs_split, embeddings)

    os.makedirs(index_dir, exist_ok=True)
    store.save_local(index_dir)

if __name__ == "__main__":
    ingest_documents()
