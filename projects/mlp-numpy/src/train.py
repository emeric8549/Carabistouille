import numpy as np
from utils import get_moons, get_circles, get_data
from model import MLP
from losses import BCE_loss, cross_entropy_loss

epochs = 1000
lr = 0.0001

#X, y = get_moons()
X, y = get_data()
input_dim = X.shape[1]
output_dim = y.shape[1]
hidden_dims = [16, 16]

model = MLP(input_dim, hidden_dims, output_dim)
criterion = BCE_loss if output_dim==1 else cross_entropy_loss

for epoch in range(epochs):
    out = model.forward(X)
    loss = criterion(y, out)

    model.backward(y, out, learning_rate=lr)

    if (epoch+1) % 100 == 0:
        pred = np.rint(out) if output_dim==1 else np.argmax(out, axis=1)
        target = y if output_dim==1 else np.argmax(y, axis=1)
        acc = np.mean(pred == target)
        print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Acc: {acc:2%}")
