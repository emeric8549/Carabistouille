from dataset import *
from preprocessing import build_vocab
from model_cbow import CBOWModel
from model_skipgram import SkipGramModel
from train import train
from utils import visualize_embeddings

corpus = [
    "le chat dort sur le canapé",
    "le chien joue dans le jardin",
    "le chat et le chien mangent",
    "le jardin est vert",
]

skipgram=False

pairs = generate_pairs(corpus, window_size=2, skipgram=skipgram)
word2idx, idx2word, vocab_size = build_vocab(pairs, skipgram)
encoded_pairs = encode_pairs(pairs, word2idx, skipgram)
 
model = SkipGramModel(vocab_size, embedding_dim=10) if skipgram else CBOWModel(vocab_size, embedding_dim=10)
train(model, encoded_pairs, epochs=100, lr=1e-1, skipgram=skipgram)

filename = "embeddings_skipgram.png" if skipgram else "embeddings_cbow.png"
#visualize_embeddings(model.W1, idx2word, filename=filename)