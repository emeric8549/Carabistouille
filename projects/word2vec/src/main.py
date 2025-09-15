import argparse
from dataset import *
from preprocessing import build_vocab
from model_cbow import CBOWModel
from model_skipgram import SkipGramModel
from train import train
from utils import visualize_embeddings
from data_utils import load_wiki_texts

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a Word2Vec model (CBOW or Skipgram) on a wikipedia dataset")
    parser.add_argument("--skipgram", type=bool, default=True, help="Use the Skipgram model (CBOW otherwise)")
    parser.add_argument("--window_size", type=int, default=2, help="Size of the window for context")
    parser.add_argument("--emb_size", type=int, default=50, help="Size of the embeddings")
    parser.add_argument("--epochs", type=int, default=100, hel="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for training")

    args = parser.parse_args()

    skipgram = args.skipgram

    sentences = list(load_wiki_texts('data/frwiki-latest-pages-articles.xml.bz2', limit=10))
    pairs = generate_pairs(" ".join(sum(sentences, [])), window_size=args.window_size, skipgram=skipgram)
    #pairs = generate_pairs(corpus, window_size=2, skipgram=skipgram)

    word2idx, idx2word, vocab_size = build_vocab(pairs, skipgram)
    encoded_pairs = encode_pairs(pairs, word2idx, skipgram)
 
    model = SkipGramModel(vocab_size, embedding_dim=args.emb_size) if skipgram else CBOWModel(vocab_size, embedding_dim=args.emb_size)
    train(model, encoded_pairs, epochs=args.epochs, lr=args.lr, skipgram=skipgram)

    filename = "embeddings_skipgram.png" if skipgram else "embeddings_cbow.png"
    visualize_embeddings(model.W1, idx2word, filename=filename)