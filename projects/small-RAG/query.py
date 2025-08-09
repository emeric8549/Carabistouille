from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})
llm = OpenAI(temperature=0) # Need to add API key

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

while True:
    query = input("Ask your question about physics (or 'quit'): ")
    if query.lower() == "quit":
        break
    print(qa.run(query))