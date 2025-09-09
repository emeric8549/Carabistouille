import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})
llm = OpenAI(temperature=0, openai_api_key=api_key) # Need to add API key

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

while True:
    query = input("Ask your question about physics (or 'quit'): ")
    if query.lower() == "quit":
        break
    print(qa.run(query))