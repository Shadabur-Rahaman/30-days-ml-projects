import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def standardize_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def apply_pca(X_scaled, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca.explained_variance_ratio_

def plot_variance_ratio(explained_variance_ratio):
    plt.figure(figsize=(6, 4))
    plt.plot(np.cumsum(explained_variance_ratio), marker='o')
    plt.title('Explained Variance by Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/explained_variance_ratio.png")
    plt.show()

def plot_pca_2d(X_pca, labels):
    plt.figure(figsize=(6, 5))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=str(label), alpha=0.7)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA - 2D Projection')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/pca_2d_projection.png")
    plt.show()
