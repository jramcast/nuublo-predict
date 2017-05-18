"""
Module for preprocessing data before
feeding it into the classfier
"""
import string
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.feature_extraction import DictVectorizer
from textblob import TextBlob

STOPWORDS = stopwords.words('english')


def tokenize(text):
    """
    Tokenizes a text
    :return: list of tokens
    """
    non_words = list(string.punctuation)
    non_words.extend(['¿', '¡'])
    text = ''.join([c for c in text if c not in non_words])
    sentence = TextBlob(text)
    tokens = [word.lemmatize() for word in sentence.words]
    return tokens


class SentimentExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts sentiment features from tweets
    """

    def __init__(self):
        pass

    def transform(self, tweets, y_train=None):
        samples = []
        for tweet in tweets:
            textBlob = TextBlob(tweet)
            samples.append({
                'sent_polarity': textBlob.sentiment.polarity,
                'sent_subjetivity': textBlob.sentiment.subjectivity
            })
        vectorized = DictVectorizer().fit_transform(samples).toarray()
        vectorized = Imputer().fit_transform(vectorized)
        vectorized_scaled = MinMaxScaler().fit_transform(vectorized)
        return vectorized_scaled

    def fit(self, X, y=None):
        return self


class TempExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts weather temp from tweet
    """
    def transform(self, tweets, y_train=None):
        tempetures = [[self.get_temperature(tweet)] for tweet in tweets]
        vectorized = self.imputer.transform(tempetures)
        vectorized_scaled = MinMaxScaler().fit_transform(vectorized)
        return vectorized_scaled

    def fit(self, tweets, y=None):
        self.imputer = Imputer()
        tempetures = [[self.get_temperature(tweet)] for tweet in tweets]
        self.imputer.fit(tempetures)
        return self

    def get_temperature(self, tweet):
        match = re.search(r'(\d+(\.\d)?)\s*F', tweet, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            celsius = (value - 32) / 1.8
            if - 100 < celsius < 100:
                return celsius
        return None


class WindExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts wind from tweet
    """

    def transform(self, tweets, y_train=None):
        winds = [[self.get_wind(tweet)] for tweet in tweets]
        vectorized = self.imputer.transform(winds)
        vectorized_scaled = MinMaxScaler().fit_transform(vectorized)
        return vectorized_scaled

    def fit(self, tweets, y=None):
        self.imputer = Imputer()
        winds = [[self.get_wind(tweet)] for tweet in tweets]
        self.imputer.fit(winds)
        return self

    def get_wind(self, tweet):
        match = re.search(r'(\d+(\.\d)?)\s*mph', tweet, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            kph = value * 1.60934
            if 0 <= kph < 500:
                return kph
        return None
