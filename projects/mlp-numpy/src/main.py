import numpy as np
from utils import get_moons, get_circles, get_data, split_data, draw_boundary
from model import MLP
from train import train
from losses import BCE_loss, cross_entropy_loss
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)


dataset_name = config["data"]["dataset"]
if dataset_name == "get_moons" or dataset_name == "get_circles":
    dataset = get_moons if dataset_name=="get_moons" else get_circles
    X, y = dataset(n_samples=int(config["data"]["n_samples"]),
                   noise=float(config["data"]["noise"]),
                   seed=int(config["data"]["seed"]))

elif dataset_name == "get_data":
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
patience = config["model"]["patience"]

model = MLP(input_dim, hidden_dims, output_dim)
criterion = BCE_loss if output_dim==1 else cross_entropy_loss


best_model = train(model, X_train, y_train, X_test, y_test, epochs, criterion, lr, patience)


if dataset_name == "get_moons" or dataset_name == "get_circles":
    filename = "decision_boundary_" + dataset_name[4:] + config["data"]["noise"] + ".png"
    draw_boundary(X_test, y_test, best_model, filename)