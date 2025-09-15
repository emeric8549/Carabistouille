import numpy as np

def train(model, encoded_pairs, epochs=10, lr=0.01, skipgram=False):
    print("Starting training...")
    if skipgram:
        loss_fn = lambda y_true, y_pred: -np.sum(np.log(y_pred[y_true] + 1e-9)) / len(y_true)
    else:
        loss_fn = lambda y_true, y_pred: -np.log(y_pred[y_true] + 1e-9)
    for epoch in range(epochs):
        total_loss = 0
        for source, y_true in encoded_pairs:
            y_pred, hidden = model.forward(source)
            loss = loss_fn(y_true, y_pred)
            total_loss += loss

            model.backward(source, y_true, y_pred, hidden, lr=lr)

        avg_loss = total_loss / len(encoded_pairs)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")