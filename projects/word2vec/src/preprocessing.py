def tokenize(sentence):
    tokenized = sentence.split(" ")
    return [word.lower() for word in tokenized]

def build_vocab(pairs):
    vocab = set()
    word2idx, idx2word = {}, {}

    for context, target in pairs:
        vocab.update(context)
        vocab.update(target)

    for i, word in enumerate(vocab):
        word2idx[word] = i
        idx2word[i] = word

    return word2idx, idx2word, len(vocab)