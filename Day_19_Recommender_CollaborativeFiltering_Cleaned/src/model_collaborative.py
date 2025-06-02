from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_similarity(matrix, kind='user'):
    if kind == 'user':
        return cosine_similarity(matrix.fillna(0))
    else:
        return cosine_similarity(matrix.T.fillna(0))

def get_recommendations(user_id, matrix, similarity_matrix, top_n=5):
    user_ratings = matrix.loc[user_id]
    scores = similarity_matrix[user_id].dot(matrix.fillna(0)) / similarity_matrix[user_id].sum()
    recommendations = scores.sort_values(ascending=False).head(top_n)
    return recommendations
