import nltk
from nltk.corpus import brown, reuters, gutenberg
import numpy as np

def get_data(corpus_name):
    if corpus_name == "brown":
        nltk.download("brown")
        sentences = brown.sents()
    elif corpus_name == "reuters":
        nltk.download('punkt_tab')
        nltk.download("reuters")
        sentences = reuters.sents()
    elif corpus_name == "gutenberg":
        nltk.download("gutenberg")
        sentences = gutenberg.sents()
    else:
        raise ValueError(f"Unknown dataset: {corpus_name}")

    return sentences


def generate_pairs(corpus, window_size=2, skipgram=False):
    corpus_pairs = []
    for sentence in corpus:
        pairs = []
        for i in range(len(sentence)):
            start = max(0, i - window_size)
            end = min(len(sentence), i + window_size + 1)

            context = [sentence[j] for j in range(start, end) if j != i]
            target = sentence[i]

            if skipgram:
                pairs.append([target, context])
            else:
                pairs.append([context, target])

        corpus_pairs.extend(pairs)

    return corpus_pairs


def build_vocab(pairs, skipgram=False):
    vocab = set()
    word2idx, idx2word = {}, {}

    for source, target in pairs:
        if skipgram:
            vocab.add(source)
            vocab.update(target)
        else:
            vocab.update(source)
            vocab.add(target)

    word2idx["pad_token"] = 0
    idx2word[0] = "pad_token"

    for i, word in enumerate(sorted(vocab), start=1):
        word2idx[word] = i
        idx2word[i] = word

    return word2idx, idx2word, len(vocab) + 1


def encode_pairs(pairs, word2idx, skipgram=False):
    encoded = []
    for source, target in pairs:
        if skipgram:
            source_id = word2idx[source]
            target_id = [word2idx[word] for word in target]
        else:
            source_id = [word2idx[word] for word in source]
            target_id = word2idx[target]
        encoded.append([source_id, target_id])

    return encoded


class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def pad_sequences(sequences, max_len, pad_value=0):
    padded = np.full((len(sequences), max_len), pad_value, dtype=np.int64)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]

    return padded


class Dataloader:
    def __init__(self, dataset, batch_size=32, shuffle=True, window_size=2, skipgram=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.window_size = window_size
        self.skipgram = skipgram
        self.indices = np.arange(len(self.dataset))
        self.max_len = 2 * window_size


    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for start_idx in range(0, len(self.dataset), self.batch_size):
            batch_idx = self.indices[start_idx:start_idx + self.batch_size]
            batch = [self.dataset[i] for i in batch_idx]
            X_batch, y_batch = zip(*batch)
            if self.skipgram:
                X_batch = np.array(X_batch)
                y_batch = pad_sequences(y_batch, self.max_len)
            else:
                X_batch = pad_sequences(X_batch, self.max_len)
                y_batch = np.array(y_batch)
                
            yield np.array(X_batch), np.array(y_batch)