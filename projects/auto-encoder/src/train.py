from tqdm import tqdm
import torch
from torcheval.metrics import PeakSignalNoiseRatio

def test(model, dataloader_test, device):
    model.eval()
    model.to(device)
    psnr = PeakSignalNoiseRatio()
    psnr_list = []
    mse_list = []
    with torch.no_grad():
        for x, y in dataloader_test:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            psnr_list.append(psnr.update(output, y).compute().item())
            mse_list.append(torch.nn.functional.mse_loss(output, y).item())
    print(f"Test loss: {sum(mse_list) / len(mse_list)}")
    print(f"PSNR: {sum(psnr_list) / len(psnr_list)}")

def train(model, dataloader_train, dataloader_test, optimizer, criterion, n_epochs, device):
    model.train()
    model.to(device)
    for e in range(n_epochs):
        for x, y in tqdm(dataloader_train, desc=f"Iteration {e+1}"):
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        print(f"Train loss: {loss.item()}")
        test(model, dataloader_test, device)