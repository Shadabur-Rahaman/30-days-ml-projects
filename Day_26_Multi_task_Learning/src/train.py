import torch
from torch import nn
from torch.utils.data import DataLoader
from src.utils import plot_losses

def train_model(model, train_loader, optimizer, criterion1, criterion2, device, num_epochs=20):
    model.to(device)
    model.train()

    history = {"loss_task1": [], "loss_task2": []}

    for epoch in range(num_epochs):
        epoch_loss1, epoch_loss2 = 0, 0

        for X, y1, y2 in train_loader:
            X, y1, y2 = X.to(device), y1.to(device), y2.to(device)

            optimizer.zero_grad()
            out1, out2 = model(X)

            loss1 = criterion1(out1, y1)
            loss2 = criterion2(out2.squeeze(), y2)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()

        avg_loss1 = epoch_loss1 / len(train_loader)
        avg_loss2 = epoch_loss2 / len(train_loader)

        history["loss_task1"].append(avg_loss1)
        history["loss_task2"].append(avg_loss2)

        print(f"Epoch {epoch+1}: Task1 Loss = {avg_loss1:.4f}, Task2 Loss = {avg_loss2:.4f}")

    plot_losses(history)
    return model, history
