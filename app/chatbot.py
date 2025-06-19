import os
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def main():
    # Load embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.load_local("faiss_index", embeddings)

    # Setup retriever and QA chain
    retriever = store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        retriever=retriever
    )

    print("Chatbot ready! Type your question or 'exit' to quit.")

    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        answer = qa_chain.run(query)
        print(f"Bot: {answer}")

if __name__ == "__main__":
    main()
