from tqdm import tqdm
import torch
from torcheval.metrics import PeakSignalNoiseRatio #, StructuralSimilarity

def test(model, dataloader_test, device):
    model.eval()
    model.to(device)
    psnr = PeakSignalNoiseRatio()
    # ssim = StructuralSimilarity()
    psnr_list = []
    # ssim_list = []
    mse_list = []
    with torch.no_grad():
        for x, y in dataloader_test:
            x = x[0].to(device)
            y = y[0].to(device)
            output = model(x)
            psnr_list.append(psnr.update(output, y).compute().item())
            # ssim_list.append(ssim(output, y).item())
            mse_list.append(torch.nn.functional.mse_loss(output, y).item())
    print(f"Test loss: {sum(mse_list) / len(mse_list)}")
    print(f"PSNR: {sum(psnr_list) / len(psnr_list)}")
    # print(f"SSIM: {sum(ssim_list) / len(ssim_list)}")

def train(model, dataloader_train, dataloader_test, optimizer, criterion, n_epochs, device):
    model.train()
    model.to(device)
    for e in range(n_epochs):
        for x, y in tqdm(dataloader_train, desc=f"Iteration {e+1}"):
            x = x[0].to(device)
            y = y[0].to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        print(f"Train loss: {loss.item()}")
        test(model, dataloader_test, device)