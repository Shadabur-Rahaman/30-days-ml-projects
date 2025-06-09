import json
import matplotlib.pyplot as plt

def save_history(history, path):
    with open(path, 'w') as f:
        json.dump(history, f)

def plot_losses(history):
    plt.figure(figsize=(10, 4))
    plt.plot(history['loss_task1'], label='Task 1 Loss')
    plt.plot(history['loss_task2'], label='Task 2 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Multi-Task Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/loss_curves_task1_task2.png')
    plt.show()


def evaluate_predictions(y_true, y_pred, task='classification'):
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
    if task == 'classification':
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred, average='weighted')
        }
    else:
        return {
            "mse": mean_squared_error(y_true, y_pred)
        }
