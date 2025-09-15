# Small RAG (Retrieval-Augmented Generation)

This project demonstrates how to build a simple Retrieval-Augmented Generation (RAG) system using `LangChain` and `Chroma` as the vector database.

## Introduction to RAG

Retrieval-Augmented Generation (RAG) is a technique that enhances the capabilities of Large Language Models (LLMs) by allowing them to retrieve relevant information from an external knowledge base before generating a response. This addresses common LLM limitations such as factual inaccuracies (hallucinations) and outdated information, providing more accurate, up-to-date, and contextually rich answers.

## Project Goal

The primary goal of this project is to illustrate the end-to-end process of building a RAG system. We will set up a knowledge base, populate it with relevant documents, and then integrate it with an LLM to answer user queries effectively.

## Core Technologies

   **LangChain**: A framework designed to simplify the creation of applications with LLMs.
   **Chroma**: A powerful and easy-to-use vector database for storing and querying embeddings.
   **HuggingFaceEmbeddings**: Used to transform text chunks into dense vector representations (embeddings).
   **CTransformers / Google Generative AI**: For integrating with various LLMs. Specifically, we demonstrate using local models (like Mistral via `CTransformers`) and cloud-based models (like Gemini via `ChatGoogleGenerativeAI`).

## How it Works

The RAG process implemented in this project follows these key steps:

1.  **Data Fetching** (`get_data.py`): Relevant information is collected from external sources. For this project, we use the Wikipedia API to download french physics-related articles.
2.  **Document Chunking** (`ingest.py`): The fetched articles are split into smaller, manageable chunks. This is crucial for efficient embedding and retrieval, as smaller chunks often contain more focused information.
3.  **Embedding** (`ingest.py`): Each text chunk is converted into a numerical vector (an embedding) using a pre-trained embedding model (e.g., `HuggingFaceEmbeddings` like `all-MiniLM-L6-v2`). These embeddings capture the semantic meaning of the text.
4.  **Vector Database Storage** (`ingest.py`): The generated embeddings, along with their corresponding text chunks, are stored in a vector database (`Chroma`). This database allows for quick similarity searches.
5.  **LLM Integration** (`query.py` or `query_local.py`): When a user asks a question, the query is also embedded. The vector database is then queried to find the most semantically similar chunks. These retrieved chunks are then provided as context to an LLM, enabling it to generate an informed and relevant response.

This setup allows the LLM to 'look up' information, significantly improving its ability to provide accurate and relevant answers based on the provided knowledge base.  


To use this project, you first need to run `get_data.py` and `ingest.py`. Then you just have to choose `query.py` if you want to use an API to access the LLM (you have to provide an API key from Google AI studio in a `.env` file) or `query_local.py` if you prefer to download a model directly on your computer. 