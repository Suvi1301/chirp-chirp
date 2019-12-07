import json
import numpy as np
import logging
import matplotlib.pyplot as plt
import os

from keras.models import model_from_json
from keras.preprocessing import image
from moviepy.editor import AudioFileClip


LOG = logging.getLogger('waitress')
LOG.setLevel(logging.INFO)

MODEL_NAME = os.getenv('CHIRP_CHOSEN_MODEL')
MODEL_PATH = f'./cnn/models/{MODEL_NAME}/'

MODEL_PARAMS = dict()
SPECTROGRAM_PARAMS = dict()
MODEL_CLASSES = dict()


def _load_spectrogram_params():
    ''' Loads the spectrogram parameters '''
    global SPECTROGRAM_PARAMS
    try:
        with open(f'spectrogram_params.json') as json_file:
            SPECTROGRAM_PARAMS = json.load(json_file)
        LOG.info('Successfully loaded Spectrogram parameters')
    except Exception as ex:
        LOG.error(f'Failed to read spectrogram params. Reason="{ex}"')


def _spectrogram(filename: str, path: str):
    ''' Generates a Spectrogram from an mp3 and saves it'''
    try:
        audio = AudioFileClip(f'{path}/{filename}.mp3')
        audio_data = audio.to_soundarray()
        audio_data = audio_data[:, 0]
        fig = plt.figure()
        plt.specgram(
            audio_data,
            Fs=SPECTROGRAM_PARAMS['FRAME_RATE'],
            NFFT=SPECTROGRAM_PARAMS['NFFT'],
            window=np.hamming(512),
            cmap='inferno',
        )
        fig.savefig(f'{path}/{filename}.jpg')
        plt.close(fig)
        LOG.info(f'Successfully saved Spectrogram {filename}.jpg to {path}')
    except Exception as ex:
        LOG.error(
            f'Failed to convert {filename} to Spectrogram. Reason="{ex}"'
        )


def _load_classes_idx():
    ''' Loads the class indices for the model '''
    global MODEL_CLASSES
    try:
        with open(f'{MODEL_PATH}{MODEL_NAME}_classes.json') as json_file:
            MODEL_CLASSES = json.load(json_file)
            MODEL_CLASSES = {
                value: key for key, value in MODEL_CLASSES.items()
            }
        LOG.info('Successfully loaded class indices')
    except Exception as ex:
        LOG.error(f'Failed to read class indices. Reason="{ex}"')


def _load_model_params():
    ''' Loads the model parameters '''
    global MODEL_PARAMS
    try:
        with open(f'{MODEL_PATH}{MODEL_NAME}_params.json') as json_file:
            MODEL_PARAMS = json.load(json_file)
        LOG.info(f'Successfully loaded the model parameters for "{MODEL_NAME}"')
    except Exception as ex:
        LOG.error(f'Failed to read model params. Reason="{ex}"')


def _load_model():
    ''' Loads the 'chosen' model '''
    try:
        json_file = open(f'{MODEL_PATH}{MODEL_NAME}.json', 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights(f'{MODEL_PATH}{MODEL_NAME}.h5')
        model.compile(optimizer=MODEL_PARAMS['OPTIMIZER'], loss=MODEL_PARAMS['LOSS_FUNC'], metrics=['accuracy', 'categorical_accuracy'])
        LOG.info(f'Loaded {MODEL_NAME} model')
        return model
    except Exception as ex:
        LOG.error(f'Cannot open {MODEL_NAME} file. Reason="{ex}"')


def _classify(model, filename: str, path: str):
    try:
        img = image.load_img(f'{path}/{filename}.jpg', target_size=tuple(MODEL_PARAMS['INPUT_SHAPE']))
        img = image.img_to_array(img)
        img = img / 255
        img = np.expand_dims(img, axis=0)
        pred_class_idx = model.predict_classes(img)
        pred_prob = model.predict_proba(img)
        result = {
            'class': MODEL_CLASSES[pred_class_idx[0]],
            'prob': pred_prob[0][pred_class_idx[0]],
        }
        return result
    except Exception as ex:
        LOG.error(f'Error classifying. Reason="{ex}"')


def cleanup(filename: str, path: str):
    os.system(f'rm {path}/{filename}.*')
    LOG.info(f'Deleted uploaded files: {filename}')


def predict_one(filename: str, path: str):
    ''' Uses the model to predict a single input file '''
    _load_spectrogram_params()
    _spectrogram(filename, path)
    _load_model_params()
    _load_classes_idx()
    model = _load_model()
    result = _classify(model, filename, path)
    return result


def predict_batch():
    # TODO: Write method to batch test data and return metrics for model
    pass


def main():
    # TODO: For testing the model with batch test data
    pass

if __name__ == "__main__":
    main()