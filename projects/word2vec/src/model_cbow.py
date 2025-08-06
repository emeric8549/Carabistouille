import numpy as np

class CBOWModel:
    def __init__(self, vocab_size, embedding_dim):
        self.W1 = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W2 = np.random.randn(embedding_dim, vocab_size) * 0.01

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def forward(self, context_ids):
        context_vecs = self.W1[context_ids]
        hidden = np.mean(context_vecs, axis=0)
        scores = np.dot(hidden, self.W2)
        probs = self.softmax(scores)

        return probs, hidden

    def backward(self, context_ids, target_id, probs, hidden, lr=1e-3):
        dscores = probs.copy()
        dscores[target_id] -= 1
        dW2 = np.outer(hidden, dscores)
        dhidden = self.W2 @ dscores
        dW1 = np.zeros_like(self.W1)
        for i in context_ids:
            dW1[i] += dhidden / len(context_ids)

        self.W1 -= lr * dW1
        self.W2 -= lr * dW2