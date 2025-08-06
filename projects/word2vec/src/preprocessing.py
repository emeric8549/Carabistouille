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

    for i, word in enumerate(vocab):
        word2idx[word] = i
        idx2word[i] = word

    return word2idx, idx2word, len(vocab)