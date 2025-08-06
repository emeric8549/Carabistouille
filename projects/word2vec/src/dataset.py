from preprocessing import tokenize


def generate_pairs(corpus, window_size=2, skipgram=False):
    corpus_pairs = []
    for sentence in corpus:
        tokenized = tokenize(sentence)

        pairs = []
        for i in range(len(tokenized)):
            start = max(0, i - window_size)
            end = min(len(tokenized), i + window_size + 1)

            context = [tokenized[j] for j in range(start, end) if j != i]
            target = tokenized[i]

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