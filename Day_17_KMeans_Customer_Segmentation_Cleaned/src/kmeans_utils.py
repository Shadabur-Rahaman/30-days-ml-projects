import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def load_and_scale_data(csv_path):
    df = pd.read_csv(csv_path)
    features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return df, scaled

def plot_elbow_curve(data, max_k=10):
    inertia = []
    for k in range(1, max_k+1):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(data)
        inertia.append(km.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k+1), inertia, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.grid()
    plt.savefig('images/elbow_method_plot.png')
    plt.close()

def plot_clusters_2d(data, labels):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[:, 0], y=data[:, 2], hue=labels, palette='Set2', s=100)
    plt.xlabel('Age')
    plt.ylabel('Spending Score')
    plt.title('Customer Segments (2D)')
    plt.legend(title='Cluster')
    plt.savefig('images/customer_segments_2D.png')
    plt.close()

def plot_clusters_3d(data, labels):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='Set2', s=50)
    ax.set_xlabel('Age')
    ax.set_ylabel('Annual Income (k$)')
    ax.set_zlabel('Spending Score')
    plt.title('Customer Segments (3D)')
    plt.savefig('images/customer_segments_3D.png')
    plt.close()
