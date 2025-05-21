import numpy as np
from model import MLP
from losses import BCE_loss, cross_entropy_loss

x = np.random.randn(4, 2)
y = np.array(([1, 0], [1, 0], [0, 1], [0, 1]))
model = MLP(input_dim=2, hidden_dims=[16, 16], output_dim=2)

y_pred = model.forward(x)

loss = cross_entropy_loss(y, y_pred)
print(loss)

model.backward(y, y_pred)
