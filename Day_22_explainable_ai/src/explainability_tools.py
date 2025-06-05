import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import tensorflow as tf
import cv2

def explain_tabular_shap(model, X, feature_names, class_names):
    explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)
    shap.plots.bar(shap_values, max_display=10)
    shap.plots.waterfall(shap_values[0])
    return shap_values

def explain_tabular_lime(model, X_train, sample, feature_names, class_names):
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    explanation = explainer.explain_instance(sample, model.predict_proba)
    explanation.as_pyplot_figure()
    return explanation

def gradcam(model, img_array, layer_name="conv_layer"):
    """Compute Grad-CAM heatmap for CNN models"""
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)[0]

    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
