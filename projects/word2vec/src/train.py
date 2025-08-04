import numpy as np
from model_cbow import CBOWModel

def train(model, encoded_pairs, epochs=10, lr=0.01):
    for epoch in range(epochs):
        total_loss = 0
        for context_ids, target_id in encoded_pairs:
            probs, hidden = model.forward(context_ids)
            loss = -np.log(probs[target_id] + 1e-9)
            total_loss += loss

            model.backward(target_id, context_ids, probs, hidden, lr=lr)

        avg_loss = total_loss / len(encoded_pairs)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")