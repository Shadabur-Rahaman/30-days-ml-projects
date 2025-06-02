import pandas as pd

def load_and_merge_data():
    ratings = pd.read_csv("data/ratings.csv")
    movies = pd.read_csv("data/movies.csv")
    return pd.merge(ratings, movies, on="movieId")

def create_user_item_matrix(merged_df):
    return merged_df.pivot_table(index='userId', columns='title', values='rating')
