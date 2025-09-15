import numpy as np

class CBOWModel:
    def __init__(self, vocab_size, embedding_dim):
        self.W1 = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W2 = np.random.randn(embedding_dim, vocab_size) * 0.01

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, context_ids):
        context_vecs = self.W1[context_ids]
        hidden = np.mean(context_vecs, axis=1)
        scores = np.dot(hidden, self.W2)
        probs = self.softmax(scores)

        return probs, hidden

    def backward(self, context_ids, target_ids, probs, hidden, lr=1e-3):
        batch_size = context_ids.shape[0]
        max_len = context_ids.shape[1]

        dscores = probs.copy()
        dscores[np.arange(batch_size), target_ids] -= 1
        dscores /= batch_size # Average over batch

        dW2 = hidden.T @ dscores

        dhidden = dscores @ self.W2.T

        dW1 = np.zeros_like(self.W1)
        for i in range(batch_size):
            for j in context_ids[i]:
                if j != 0:  # Ignore padding
                    continue
                dW1[j] += dhidden[i] / max_len

        self.W1 -= lr * dW1
        self.W2 -= lr * dW2