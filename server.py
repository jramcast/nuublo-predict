#!/usr/bin/env python

import sys
from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from json import dumps
from flask_jsonpify import jsonify
from classify.classes import CLASSES
from classify import classifier
from textblob import TextBlob
from textblob.exceptions import TranslatorError


app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()
parser.add_argument('text')
classifier.load()


class Prediction(Resource):
    """
    REST resource: /prediction
    """

    def get(self):
        """
        GET /prediction
        """
        result = {'info': 'Prediction endpoint. Make a POST request to make a prediction'}
        return jsonify(result)

    def post(self):
        """
        POST /prediction
        """
        args = parser.parse_args()
        try:
            text = translate_to_en(args['text'])
        except TranslatorError:
            return {
                'error': 'Translation error: {}'.format(sys.exc_info()[0])
            }, 500
        result = classifier.predict([text])
        return format_result(result), 201

api.add_resource(Prediction, '/prediction')


def translate_to_en(text):
    """
    Translate any text into english
    """
    # Remove hash because it affects translation
    blob = TextBlob(text.replace('#', ''))
    if blob.detect_language() != 'en':
        text = blob.translate(to='en').raw
    return text


def format_result(result):
    """
    Formats the classification result to send to the client
    """
    formatted = {}
    for classname in result:
        if result[classname] == 1:
            formatted[classname] = CLASSES[classname]
    return {
        'classes': formatted,
    }


if __name__ == '__main__':
    app.run(port='5002')
