import argparse
from dataset import get_data, generate_pairs, encode_pairs, Dataset, Dataloader
from preprocessing import build_vocab
from model_cbow import CBOWModel
from model_skipgram import SkipGramModel
from train import train
from utils import visualize_embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Word2Vec model (CBOW or Skipgram) on a wikipedia dataset")
    parser.add_argument("--corpus", type=str, default="brown", choices=["brown", "reuters", "gutenberg"], help="Corpus used to train the model")
    parser.add_argument("--model", type=str, default="cbow", choices=["cbow", "sg"], help="Model to use (CBOW or Skipgram)")
    parser.add_argument("--window_size", type=int, default=2, help="Size of the window for context")
    parser.add_argument("--emb_size", type=int, default=20, help="Size of the embeddings")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for training")

    args = parser.parse_args()

    skipgram = True if args.model == "sg" else False

    sentences = get_data(args.corpus)
    pairs = generate_pairs(sentences, window_size=args.window_size, skipgram=skipgram)
    word2idx, idx2word, vocab_size = build_vocab(pairs, skipgram)
    encoded_pairs = encode_pairs(pairs, word2idx, skipgram)
    dataset = Dataset(*zip(*encoded_pairs))
    dataloader = Dataloader(dataset, batch_size=32, shuffle=True, skipgram=skipgram)

    model = SkipGramModel(vocab_size, embedding_dim=args.emb_size) if skipgram else CBOWModel(vocab_size, embedding_dim=args.emb_size)
    train(model, dataloader, epochs=args.epochs, lr=args.lr, skipgram=skipgram)

    filename = "embeddings_skipgram.png" if skipgram else "embeddings_cbow.png"
    visualize_embeddings(model.W1, idx2word, filename=filename)