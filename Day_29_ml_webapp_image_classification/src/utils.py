import matplotlib.pyplot as plt

def plot_metrics(history, output_path='outputs/training_plot.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_path)
    plt.close()
