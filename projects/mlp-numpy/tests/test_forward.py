import numpy as np
from model import MLP

x = np.random.randn(4, 2)
print(x)
model = MLP(input_dim=2, hidden_dims=[16, 16], output_dim=1)
pred = model.predict(x)
print(pred)
