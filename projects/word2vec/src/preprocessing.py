def tokenize(sentence):
    return sentence.split(" ")

def build_vocab(pairs):
    vocab = set()
    word2idx, idx2word = {}, {}

    for context, target in pairs:
        vocab.update(context)
        vocab.add(target)

    for i, word in enumerate(vocab):
        word2idx[word] = i
        idx2word[i] = word

    return word2idx, idx2word, len(vocab)