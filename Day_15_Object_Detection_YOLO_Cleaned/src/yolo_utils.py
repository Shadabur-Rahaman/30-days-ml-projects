import cv2
import torch
from matplotlib import pyplot as plt

def load_model(weights_path='models/yolov5s.pt'):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    return model

def run_inference(model, image_path):
    results = model(image_path)
    return results

def plot_results(results):
    results.print()
    results.show()
    results.save()  # Saves in 'runs/detect/exp'

def visualize_from_results(results):
    for img in results.ims:
        plt.imshow(img)
        plt.axis('off')
        plt.show()
