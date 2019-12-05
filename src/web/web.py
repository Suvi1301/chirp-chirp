import os
import urllib.request

from flask import Flask, request, redirect, jsonify

APP = Flask(__name__)

@APP.route('/test')
def hello():
    return jsonify({'about': 'Hello World'})


if __name__ == "__main__":
    APP.run()
