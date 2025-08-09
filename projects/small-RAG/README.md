# How to build a RAG

This small project presents how to quickly build a RAG (Retrieval-Augmented Generation) using LangChain and a Vector database.  
  
To illustrate the process, we use the Wikipedia API to download some physics articles using the `get_data.py`file.  
Once we have the corpus, we just have to cut it into several chunks that will then be embedded using a particular model. Word2Vec could work but due to its structure, we would lose the context of the chunk. A better model could be the Sentence Transformer or even BERT or its equivalents.  
After the chunk is embedded, we finally just have to store it into a Vector database such as Chroma.  

The real magic can now appear: when a user asks a question to a LLM, it can now retrieve information from the vector database to enrich its response. This process is realized by computing the distance (cosine, euclidian, …) between the query and the embedded chunks in the DB and taking the most similar ones.