import numpy as np

class SkipGramModel:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.W1 = np.random.randn(vocab_size, embedding_dim)
        self.W2 = np.random.randn(embedding_dim, vocab_size)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, target_id):
        hidden = self.W1[target_id]
        scores = hidden @ self.W2
        probs = self.softmax(scores)
        return probs, hidden

    def backward(self, target_ids, context_ids_batch, probs, hidden, lr=1e-3):
        batch_size = len(target_ids)
        dW1 = np.zeros_like(self.W1)
        dW2 = np.zeros_like(self.W2)
        
        for i in range(batch_size):
            context_ids = context_ids_batch[i]
            target_id = target_ids[i]

            context_ids = [w for w in context_ids if w != 0]  # Ignore padding
            if len(context_ids) == 0:
                continue

            error = probs[i].copy()
            error[context_ids] -= 1 / len(context_ids)  # Average over the context

            dW2 += np.outer(hidden[i], error)
            dW1[target_id] += self.W2 @ error

        self.W1 -= lr * dW1
        self.W2 -= lr * dW2