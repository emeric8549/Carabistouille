def tokenize(sentence):
    tokenized = sentence.split(" ")
    return [word.lower() for word in tokenized]

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