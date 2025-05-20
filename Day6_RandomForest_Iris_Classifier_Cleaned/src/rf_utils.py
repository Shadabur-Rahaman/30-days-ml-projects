# src/rf_utils.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    X = df.drop('species', axis=1)
    y = df['species']
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt='g')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("images/confusion_matrix.png")
    plt.close()


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_df.sort_values(by="Importance", ascending=False, inplace=True)

    sns.barplot(x="Importance", y="Feature", data=feat_df)
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig("images/feature_importance.png")
    plt.close()
