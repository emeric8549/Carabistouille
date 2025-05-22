import numpy as np
from utils import get_moons, get_circles, get_data, split_data
from model import MLP
from losses import BCE_loss, cross_entropy_loss
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)


dataset = config["data"]["dataset"]
if dataset == "get_moons" or dataset == "get_circles":
    dataset = get_moons if dataset=="get_moons" else get_circles
    X, y = dataset(n_samples=int(config["data"]["n_samples"]),
                   noise=float(config["data"]["noise"]),
                   seed=int(config["data"]["seed"]))

elif dataset == "get_data":
    X, y = get_data(n_samples=int(config["data"]["n_samples"]),
                   n_features=int(config["data"]["n_features"]),
                   n_redundant=int(config["data"]["n_redundant"]),
                   n_informative=int(config["data"]["n_informative"]),
                   n_clusters_per_class=int(config["data"]["n_clusters_per_class"]),
                   n_classes=int(config["data"]["n_classes"]),
                   seed=int(config["data"]["seed"]))

else:
    print("Dataset not available")
    exit()

X_train, y_train, X_test, y_test = split_data(X, y, float(config["data"]["test_size"]))

input_dim = X.shape[1]
output_dim = y.shape[1]
hidden_dims = config["model"]["hidden_dims"]

epochs = int(config["model"]["epochs"])
lr = float(config["model"]["learning_rate"])

model = MLP(input_dim, hidden_dims, output_dim)
criterion = BCE_loss if output_dim==1 else cross_entropy_loss

for epoch in range(epochs):
    out = model.forward(X_train)
    loss = criterion(y_train, out)

    model.backward(y_train, out, learning_rate=lr)

    if (epoch+1) % 100 == 0:
        out_test = model.forward(X_test)
        test_loss = criterion(y_test, out_test)

        pred = np.rint(out) if output_dim==1 else np.argmax(out, axis=1)
        target = y_train if output_dim==1 else np.argmax(y_train, axis=1)
        acc = np.mean(pred == target)

        pred_test = np.rint(out_test) if output_dim==1 else np.argmax(out_test, axis=1)
        target_test = y_test if output_dim==1 else np.argmax(y_test, axis=1)
        test_acc = np.mean(pred_test == target_test)

        print(f"Epoch {epoch+1} | Train loss: {loss:.4f} | Test loss: {test_loss:.4f} | Train acc: {acc:2%} | Test acc: {test_acc:2%}")
