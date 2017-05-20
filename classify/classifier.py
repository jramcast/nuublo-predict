"""
The weather classifier module
"""

import os
import time
import math
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.externals import joblib
from classify.classes import CLASSES
from classify.preprocessing import (SentimentExtractor,
                           TempExtractor,
                           WindExtractor,
                           tokenize,
                           STOPWORDS)

# Loaded models will be stored here
MODELS = {}


sentiment_extractor = SentimentExtractor()
temp_extractor = TempExtractor()
wind_extractor = WindExtractor()
vectorizer = CountVectorizer(
    min_df=10,
    max_df=0.5,
    ngram_range=(1, 3),
    max_features=10000,
    lowercase=True,
    stop_words=STOPWORDS,
    tokenizer=tokenize
)


def train(data):
    """
    Trains the classifier. Each class is trained separately and saved to disk
    """
    x_train = [row['tweet'] for row in data]
    for classname in CLASSES:
        print("--------------------------> Training " + classname)
        start_time = time.time()
        y_train = filter_class(data, classname)
        classifier = train_class(x_train, y_train)
        # save model
        joblib.dump(
            classifier,
            '{}/models/{}.pkl'.format(os.path.dirname(os.path.abspath(__file__)), classname)
        )
        print('Time elapsed:')
        print(time.time() - start_time)


def load():
    """
    Loads pretrained models from disk
    """
    print('Loading models...')
    for classname in CLASSES:
        MODELS[classname] = joblib.load('classify/models/{}.pkl'.format(classname))
    print('Models loaded')


def predict(data):
    """
    Predicts class for all models
    """
    if MODELS == {}:
        load()
    results = {}
    for classname in CLASSES:
        results[classname] = MODELS[classname].predict(data)
    return results


def train_class(x_train, y_train):
    """
    Trains a model for one class
    """
    svm = LinearSVC(C=0.3, max_iter=300, loss='hinge')
    pipeline = Pipeline([
        ('union', FeatureUnion([
            ('sentiment', sentiment_extractor),
            ('temp', temp_extractor),
            ('wind', wind_extractor),
            ('vect', vectorizer),
        ])),
        ('cls', svm),
    ])

    accuracy = cross_val_score(pipeline, x_train, y_train, scoring='accuracy', n_jobs=3)
    print('=== Accuracy ===')
    print(np.mean(accuracy))
    pipeline.fit(x_train, y_train)
    return pipeline


def filter_class(data, classname):
    """
    Returns a list of 0 or 1 value based no the presence of the given class
    """
    classes = []
    for row in data:
        value = float(row[classname])
        classes.append(value > 0.5)
    return classes
