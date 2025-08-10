from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

llm = CTransformers(
    model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    config={"max_new_tokens": 512, "temperature": 0.1}
)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

while True:
    query = input("Ask your question about physics (or 'quit'): ")
    if query.lower() == "quit":
        break
    print(qa.run(query))