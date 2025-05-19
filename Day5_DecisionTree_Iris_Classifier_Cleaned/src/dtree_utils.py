import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix

def load_and_prepare_data():
    df = sns.load_dataset('iris')
    df['species'] = df['species'].astype('category').cat.codes
    return df

def split_data(df, test_size=0.2, random_state=42):
    X = df.drop('species', axis=1)
    y = df['species']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_decision_tree(X_train, y_train, max_depth=None):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return confusion_matrix(y_test, y_pred)

def plot_confusion(conf_matrix, save_path=None):
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens')
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_feature_pair(df, save_path=None):
    sns.pairplot(df, hue="species", palette="magma")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_decision_tree_model(model, feature_names, class_names, save_path=None):
    plt.figure(figsize=(12, 8))
    plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True)
    if save_path:
        plt.savefig(save_path)
    plt.show()
