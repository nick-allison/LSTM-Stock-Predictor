# Get_News_data/sentiment_analysis.py

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

#does sentiment analysis on 'sentence' and returns dictionary output
def analyze(sentence):
    return analyzer.polarity_scores(sentence)