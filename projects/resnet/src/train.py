from tqdm import tqdm
import numpy as np
import copy

import torch
import torch.nn as nn
from torch.optim import SGD


def train(model, dataloader_train, dataloader_test, device, lr, epochs=1000, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), weight_decay=1e-4, momentum=0.9, lr=lr)
    
    best_loss = float('inf')
    best_model = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x, y in tqdm(dataloader_train, desc="Training"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            pred = model(x)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.cpu().item())

        model.eval()
        losses_test, acc_test = [], []
        with torch.no_grad():
            for x, y in tqdm(dataloader_test, desc="Testing"):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                losses_test.append(criterion(pred, y).cpu().item())
                pred_labels = pred.argmax(dim=1)
                acc_test.extend((pred_labels == y).cpu())

        if np.mean(losses_test) < best_loss:
            best_loss = np.mean(losses_test)
            best_model = copy.deepcopy(model)
            patience_counter = 0
        
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}...")
                break

        if (epoch+1) % 1 == 0:
            print(f"Epoch {epoch+1} | Train loss: {np.mean(train_losses):5f} | Test loss: {np.mean(losses_test):5f} | Test accuracy: {np.mean(acc_test):2%}")

    print(f"Best loss is {best_loss:.5f}")
    filename = "best_models/" + model.name + ".pth"
    torch.save(best_model.state_dict(), filename)

    return best_model