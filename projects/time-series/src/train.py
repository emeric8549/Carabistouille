import torch
import torch.nn as nn
import torch.optim as optim


def train(model, train_loader, test_loader, device, epochs=10, lr=1e-1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            output = model(X)
            labels = torch.max(y, 1)[1]
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        test_loss = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                labels = torch.max(y, 1)[1]
                predictions = torch.max(output, 1)[1]

                loss = criterion(output, labels)
                test_loss += loss.item()
                total += labels.size(0)
                correct =+ (predictions == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Training loss: {train_loss/len(train_loader):.4f}, Test loss: {test_loss/len(test_loader):.4f}, Test acc: {acc:.2f}%")

    return model