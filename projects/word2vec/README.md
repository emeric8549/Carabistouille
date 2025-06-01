# Implementation of the Word2Vec model

The Word2Vec model has been introduced in 2013 in the paper ([Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781)) by Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean.  
The article aims to find a method to embed words efficiently. Indeed, because ML algorithms cannot use words as such, we have to create a numerical representation, the so-called embeddings. Previously, we used to make one-hot encoded vectors, where given the dictionary of all words present in the dataset, we create a vector that has the size of vocabulary (or dictionary) with a '1' at the index of the embedded word and '0' otherwise.  
  
Working with sparse and high-dimensional vectors (~30K+ words in the vocabulary) is then quickly a burden when trying to train a model. Having a dense and low-dimensional vector to represent a word allows the models to be trained faster and have a better convergence.  
Word2Vec tackles this problem by proposing a simple yet effective architecture using the context of the words. There are two variants of the model, 'Continuous Bag-Of-Words' (CBOW) and 'Skip-gram'. The former predicts the one-hot representation of a word based on the one-hot vectors of the neigboring words in the sentence (also called a window). The latter do the opposite by predicting the neighborhood of a word based on its one-hot vector. A simple MLP with one hidden layer is used in both cases so we would be able to implement Word2Vec only with NumPy. However, we use the PyTorch library in this project for performance purposes. The final dense embeddings are obtained thanks to the hidden layer of the MLP.
  
### Work to do:
    - read the article again to understand the meaning of bag-of-words in this context
    - add a better descrition of the two variants with images
    - add that it allows a good representation in space where two words that are sementically similars will be close in the a space representation. (exemple with 2 dimensional vectors)
    - use the sampling method of skip-gram to reduce computational complexity