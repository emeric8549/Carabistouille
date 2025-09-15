from preprocessing import tokenize
import nltk
from nltk.corpus import brown, reuters, gutenberg


def get_data(corpus_name):
    if corpus_name == "brown":
        nltk.download("brown")
        sentences = brown.sents()
    elif corpus_name == "reuters":
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
        #tokenized = tokenize(sentence)

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