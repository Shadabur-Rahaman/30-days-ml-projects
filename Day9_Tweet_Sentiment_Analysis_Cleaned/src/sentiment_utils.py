# src/sentiment_utils.py

from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

def get_textblob_sentiment(text):
    """
    Returns sentiment polarity using TextBlob.
    Polarity > 0: Positive, < 0: Negative, == 0: Neutral
    """
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def get_vader_sentiment(text):
    """
    Returns sentiment label using NLTK's VADER sentiment analyzer.
    """
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"
