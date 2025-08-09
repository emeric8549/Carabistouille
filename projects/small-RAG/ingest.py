from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

loader = TextLoader("data/physics_articles.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="chroma_db")
vectordb.persist()

print(f"Vector database created with {len(chunks)} chunks.")