import numpy as np
from losses import BCE_loss, cross_entropy_loss

def train(model, X_train, y_train, X_test, y_test, epochs, criterion, lr, patience):
    patience_counter = 0
    best_model = None
    best_loss, best_acc = float('inf'), 0
    best_loss_epoch, best_acc_epoch = 0, 0

    for epoch in range(epochs):
        out = model.forward(X_train)
        loss = criterion(y_train, out)

        model.backward(y_train, out, learning_rate=lr)

        out_test = model.forward(X_test)
        test_loss = criterion(y_test, out_test)

        # Compute metrics
        binary = out.shape[1]==1
        pred = np.rint(out) if binary else np.argmax(out, axis=1)
        target = y_train if binary else np.argmax(y_train, axis=1)
        acc = np.mean(pred == target)

        pred_test = np.rint(out_test) if binary else np.argmax(out_test, axis=1)
        target_test = y_test if binary else np.argmax(y_test, axis=1)
        test_acc = np.mean(pred_test == target_test)

        # Update best metrics 
        if test_acc > best_acc:
            best_acc = test_acc
            best_acc_epoch = epoch + 1

        if test_loss < best_loss:
            best_loss = test_loss
            best_model = model
            best_loss_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter == patience:
                print(f"Early stopping at epoch {epoch+1}...")
                break
        
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1} | Train loss: {loss:.4f} | Test loss: {test_loss:.4f} | Train acc: {acc:2%} | Test acc: {test_acc:2%}")

    print(f"Best loss is {best_loss:.5f} at epoch {best_loss_epoch}")
    print(f"Best accuracy is {best_acc:.2%} at epoch {best_acc_epoch}")

    return best_model



