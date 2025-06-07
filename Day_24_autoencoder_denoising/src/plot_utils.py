import matplotlib.pyplot as plt

def plot_images(original, noisy, denoised, n=10, path="images/denoised_images.png"):
    plt.figure(figsize=(20, 6))
    for i in range(n):
        # Original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(original[i].squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(noisy[i].squeeze(), cmap='gray')
        plt.title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(denoised[i].squeeze(), cmap='gray')
        plt.title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_training_loss(history, path='images/training_history.png'):
    plt.plot(history['loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.savefig(path)
    plt.close()
