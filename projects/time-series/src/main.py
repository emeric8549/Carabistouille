import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils import load_data, get_model, visualize_filters
from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rnn", choices=["rnn", "gru", "lstm", "cnn1d"], help="rnn | gru | lstm | cnn1d")
    parser.add_argument("--train_path", type=str, default="data/mitbih_train.csv")
    parser.add_argument("--test_path", type=str, default="data/mitbih_test.csv")
    parser.add_argument("--hidden_size", type=int, default=64, help="Size of hidden dimension")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_data(args.train_path, args.test_path)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    input_size = X_train.shape[2]
    num_classes = y_train.shape[1]
    model = get_model(args.model, input_size, args.hidden_size, num_classes)
    filename = f"model_{args.model}_{args.hidden_size}.pth"

    trained_model = train(model, train_loader, test_loader, args.device, epochs=args.epochs, lr=args.lr)
    torch.save(trained_model.state_dict(), filename)
    print(f"Model saved as {filename}")

    model.load_state_dict(torch.load(filename, weights_only=True))
    conv1 = model.conv1.weight.data.cpu().numpy()
    visualize_filters(conv1)