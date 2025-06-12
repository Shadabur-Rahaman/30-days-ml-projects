import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import CNNModel
import os

def train_model():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder('data/train', transform=transform)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    model = CNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    os.makedirs('model', exist_ok=True)
    torch.save(model.state_dict(), 'model/cnn_model.pth')

if __name__ == '__main__':
    train_model()


# src/model.py
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self
