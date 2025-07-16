import numpy as np

class MLP:
    def __init__(self, input_dim, hidden_dims, output_dim):
        self.layers = []
        self.biases = []
        self.last_act = self.sigmoid if output_dim == 1 else self.softmax

        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            weight = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2. / dims[i])
            bias = np.zeros((1, dims[i+1]))
            self.layers.append(weight)
            self.biases.append(bias)


    def relu(self, x):
        return np.maximum(0, x)

    def deriv_relu(self, x):
        x = np.nan_to_num(x)
        return (x > 0).astype(float)


    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x + 1e-9))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)



    def forward(self, x):
        self.activations = [x]
        self.z_values = []
        a = x
        for W, b in zip(self.layers[:-1], self.biases[:-1]):
            z = a @ W + b
            a = self.relu(z)
            self.z_values.append(z)
            self.activations.append(a)

        z = a @ self.layers[-1] + self.biases[-1]
        self.z_values.append(z)

        return z



    def backward(self, y, y_pred, learning_rate=1e-3):
        grads_w = []
        grads_b = []

        delta = y_pred - y
        for i in reversed(range(len(self.layers))):
            a_prev = self.activations[i]
            grad_w = np.dot(a_prev.T, delta)
            grad_b = np.sum(delta, axis=0, keepdims=True)
            grads_w.insert(0, grad_w)
            grads_b.insert(0, grad_b)

            if i != 0:
                delta = np.dot(delta, self.layers[i].T) * self.deriv_relu(self.z_values[i-1])

        for i in range(len(self.layers)):
            self.layers[i] -= learning_rate * np.clip(grads_w[i], -0.1, 0.1)
            self.biases[i] -= learning_rate * np.clip(grads_b[i], -0.1, 0.1)



    def predict(self, x):
        logits = self.forward(x)
        a = np.rint(self.last_act(logits)) if self.last_act == self.sigmoid else self.last_act(logits).argmax(axis=1)
        return a
