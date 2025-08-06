import numpy as np

class SkipGramModel:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.W1 = np.random.randn(vocab_size, embedding_dim)
        self.W2 = np.random.randn(embedding_dim, vocab_size)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def forward(self, target_id):
        hidden = self.W1[target_id]
        scores = np.dot(hidden, self.W2)
        probs = self.softmax(scores)
        return probs, hidden

    def backward(self, target_id, context_ids, probs, hidden, lr=1e-3):
        y_true = np.zeros(self.vocab_size)
        for idx in context_ids:
            y_true[idx] += 1
        y_true /= len(context_ids) # Average over the context

        error = probs - y_true
        dW2 = np.outer(hidden, error)
        dW1 = np.dot(self.W2, error)

        self.W1[target_id] -= lr * dW1
        self.W2 -= lr * dW2