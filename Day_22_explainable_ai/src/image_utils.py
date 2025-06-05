import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(img_path, target_size=(128, 128)):
    """Load and preprocess an image for CNN input"""
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def plot_gradcam_overlay(img_path, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on image"""
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Grad-CAM Overlay")
    plt.show()
