import os
import urllib.request
import logging
import uuid

from cnn.predict import predict_one
from cnn.predict import cleanup
from flask import Flask, request, redirect, jsonify
from werkzeug.utils import secure_filename
from waitress import serve
from moviepy.editor import AudioFileClip


LOG = logging.getLogger('waitress')
LOG.setLevel(logging.DEBUG)

APP = Flask(__name__)
ALLOWED_EXTENSIONS = set(['mp3', 'm4a'])


def is_file_audio(filename: str):
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@APP.route('/test')
def hello():
    return jsonify({'about': 'Hello World'})


@APP.route('/predict', methods=['POST'])
def predict_bird():
    if 'audio-sample' not in request.files:
        LOG.debug('No audio-sample data in request')
        resp = jsonify({'message': 'File not found'})
        resp.status_code = 400
        return resp

    file = request.files['audio-sample']

    if file.filename == '':
        LOG.debug('No file found in request')
        resp = jsonify({'message': 'File not found'})
        resp.status_code = 400
        return resp

    if file and is_file_audio(file.filename):
        filename = str(uuid.uuid1())
        file.save(os.path.join(APP.config['UPLOAD_FOLDER'], f'{filename}.mp3'))
        LOG.info(f'Uploaded {filename}.mp3 successfully')
        try:
            result = predict_one(filename, APP.config['UPLOAD_FOLDER'])
            resp = jsonify(result)
            resp.status_code = 200
        except Exception as ex:
            LOG.error(f'Failed to predict for {filename}. Reason="{ex}"')
            resp = jsonify({'message': 'Internal Server Error'})
            resp.status_code = 500
        cleanup(filename, APP.config['UPLOAD_FOLDER'])
        return resp

    else:
        LOG.debug('Invalid file format')
        resp = jsonify({'message': 'Invalid file format'})
        resp.status_code = 400
        return resp


if __name__ == "__main__":
    APP.run()
