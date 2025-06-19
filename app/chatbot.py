import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

def main():
    # Load environment variables from .env
    load_dotenv()

    # Load the local FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Create a retriever and QA chain
    retriever = store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        retriever=retriever
    )

    print("✅ Chatbot is ready. Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        result = qa_chain.run(query)
        print(f"Bot: {result}")

if __name__ == "__main__":
    main()
