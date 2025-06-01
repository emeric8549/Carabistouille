import torch.nn as nn

class Word2Vec(nn.Module):
    def __init__(self, embedding_size, vocab_size, window_size, skipgram=False):
        super(Word2Vec, self).__init__()

        in_features = vocab_size * window_size * 2 if not skipgram else vocab_size
        out_features = vocab_size * window_size * 2 if skipgram else vocab_size

        self.fc1 = nn.Linear(in_features=in_features, out_features=embedding_size)
        self.fc2 = nn.Linear(in_features=embedding_size, out_features=out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_embedded = self.fc1(x)
        out = self.relu(x_embedded)
        out = self.fc2(out)

        return out