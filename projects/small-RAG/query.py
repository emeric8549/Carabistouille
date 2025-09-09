import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})
llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash", api_key=api_key)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

while True:
    query = input("Ask your question about physics (or 'quit'): ")
    if query.lower() == "quit":
        break
    print(qa.invoke(query)["result"])