from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

def scale_features(X):
    scaler = RobustScaler()
    return scaler.fit_transform(X), scaler

def reduce_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X), pca
