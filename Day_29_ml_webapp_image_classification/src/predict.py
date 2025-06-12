import torch
from torchvision import transforms
from PIL import Image
from src.model import CNNModel
import os

# Load model
model = CNNModel()
model.load_state_dict(torch.load('model/cnn_model.pth', map_location=torch.device('cpu')))
model.eval()

# Class names (example)
class_names = ['cat', 'dog']

def predict_image_class(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]
