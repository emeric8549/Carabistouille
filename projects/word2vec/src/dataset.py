from preprocessing import tokenize


def generate_cbow_pairs(corpus, window_size=2):
    corpus_pairs = []
    for sentence in corpus:
        tokenized = tokenize(sentence)

        pairs = []
        for i in range(len(tokenized)):
            start = max(0, i - window_size)
            end = min(len(tokenized), i + window_size + 1)

            context = [tokenized[j] for j in range(start, end) if j != i]
            target = tokenized[i]

            pairs.append([context, target])

        corpus_pairs.extend(pairs)

    return corpus_pairs


def encode_pairs(pairs, word2idx):
    encoded = []
    for context, target in pairs:
        context_ids = [word2idx[word] for word in context]
        target_id = word2idx[target]
        encoded.append([context_ids, target_id])

    return encoded



def generate_skipgram_pairs(corpus, window_size=2):
    corpus_pairs = []
    for sentence in corpus:
        tokenized = tokenize(sentence)
        
        pairs = []
        for i in range(len(tokenized)):
            start = max(0, i - window_size)
            end = min(len(tokenized), i + window_size + 1)

            context = [tokenized[j] for j in range (start, end) if j != i]
            target = tokenized[i]

            pairs.append([target, context])
        
        corpus_pairs.extend(pairs)
    
    return corpus_pairs