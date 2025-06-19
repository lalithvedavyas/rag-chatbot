import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

def load_qa_pipeline(index_dir="embeddings"):
    embeddings = OpenAIEmbeddings()
    store = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    retriever = store.as_retriever(search_type="similarity", k=4)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.3)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
