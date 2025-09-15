import numpy as np
from tqdm import tqdm
import os


def train(model, dataset, epochs=10, lr=0.01, skipgram=False, filename="model_weights/embeddings.npz"):
    os.makedirs("model_weights", exist_ok=True)
    print("Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for X_batch, y_batch in tqdm(dataset, desc=f"Epoch {epoch+1}/{epochs}"):
            y_pred, hidden = model.forward(X_batch)
            if skipgram:
                batch_loss = 0
                for i, context in enumerate(y_batch):
                    for w in context:
                        if w != 0:  # Ignore padding
                            batch_loss += -np.log(y_pred[i][w] + 1e-9)
                loss = batch_loss / len(X_batch)
            else:
                loss = -np.mean([np.log(y_pred[np.arange(len(y_batch)), y_batch] + 1e-9)])

            total_loss += loss
            n_batches += 1

            model.backward(X_batch, y_batch, y_pred, hidden, lr=lr)

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        W1 = model.W1
        W2 = model.W2
        np.savez(filename, W1=W1, W2=W2)
        
    print("Training completed.")