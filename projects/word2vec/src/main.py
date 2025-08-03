from dataset import generate_cbow_pairs
from preprocessing import build_vocab

corpus = [
    "le chat dort sur le canapé",
    "le chien joue dans le jardin",
    "le chat et le chien mangent",
    "le jardin est vert",
]


pairs = generate_cbow_pairs(corpus, window_size=2)
word2idx, idx2word, vocab_size = build_vocab(pairs)
