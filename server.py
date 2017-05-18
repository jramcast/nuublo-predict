#!/usr/bin/env python

from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from json import dumps
from flask_jsonpify import jsonify
from classify.classes import CLASSES
from classify import classifier

app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()
parser.add_argument('text')

classifier.load()

class Prediction(Resource):

    def get(self):
        result = {'info': 'Prediction endpoint. Make a POST request to make a prediction'}
        return jsonify(result)

    def post(self):
        args = parser.parse_args()
        text = args['text']
        result = classifier.predict([text])
        formatted_result = []
        for classname in result:
            if result[classname] == 1:
                formatted_result = formatted_result + [CLASSES[classname]]
        return formatted_result, 201

api.add_resource(Prediction, '/prediction') # Route_1

if __name__ == '__main__':
     app.run(port='5002')
