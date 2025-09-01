from tqdm import tqdm
import numpy as np

import torch.nn as nn
from torch.optim import SGD


def train(model, dataloader_train, dataloader_test, device, lr, epochs=1000, patience=10):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(resnet.parameters(), weight_decay=1e-4, momentum=0.9, lr=lr)
    
    best_loss = float('inf')
    best_model = None
    patience_counter = 0

    for epoch in range(epochs):
        train_losses = []
        for x, y in tqdm(dataloader_train, desc="Training"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            pred = model(x)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            
        losses_test, acc_test = [], []
        for x, y in tqdm(dataloader_test, desc="Testing"):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            losses_test.append(criterion(pred, y).item())
            pred_labels = pred.argmax(dim=1)
            acc_test.extend(pred_labels == y)

        if np.mean(losses_test) < best_loss:
            best_loss = np.mean(losses_test)
            best_model = model
            patience_counter = 0
        
        else:
            patience_counter += 1
            if patience_counter == patience:
                print(f"Early stopping at epoch {epoch+1}...")
                break

        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1} | Train loss: {np.mean(np.array(train_losses)):5f} | Test loss: {np.mean(np.array(losses_test)):5f} | Test accuracy: {np.mean(np.array(acc_test)):2%}")

    print(f"Best loss is {best_loss:.5f}")
    filename = "best_models/" + model.name + ".pth"
    torch.save(model.state_dict(), filename)

    return best_model