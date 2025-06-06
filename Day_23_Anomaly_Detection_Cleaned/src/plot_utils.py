import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def plot_score_distribution(scores):
    sns.histplot(scores, bins=50, kde=True)
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.show()
