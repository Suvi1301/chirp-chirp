import os
import json
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from docopt import docopt

TRAINING_DATA_PATH = './../data/images/training_data/'
TESTING_DATA_PATH = './../data/images/testing_data/'
IMAGE_SHAPE = (64, 64, 3)
BATCH_SIZE = 32
EPOCHS = 25
MODEL_NAME = ''
METRICS = ['accuracy', 'categorical_accuracy']
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'binary_crossentropy'
NUM_CLASSES = 20


def train_new_cnn():
    ''' Create a CNN model '''
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=IMAGE_SHAPE, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=NUM_CLASSES, activation='softmax'))

    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=METRICS)
    print(f'Model "{MODEL_NAME}" compiled')

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    training_set = train_datagen.flow_from_directory(
        TRAINING_DATA_PATH,
        target_size=(64, 64),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
    )

    save_class_indices(training_set)

    testing_set = test_datagen.flow_from_directory(
        TESTING_DATA_PATH,
        target_size=(64, 64),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
    )

    model.fit_generator(
        training_set,
        steps_per_epoch=training_set.samples / BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=testing_set,
        validation_steps=testing_set.samples / BATCH_SIZE,
    )
    print(f'Model "{MODEL_NAME}" trained')
    save_model(model)


def save_class_indices(generator):
    try:
        with open('models/{MODEL_NAME}_classes.json', 'w') as json_file:
            json.dump(generator.class_indices, json_file, indent=4)
    except Exception as ex:
        print(
            f'ERROR: Failed to save class indices for {MODEL_NAME}. Reason="{ex}"'
        )


def save_model(model):
    ''' Save trained model in JSON and H5PY '''
    model_json = model.to_json()
    try:
        with open(f'models/{MODEL_NAME}.json', 'w') as json_file:
            json_file.write(model_json)
        model.save_weights(f'models/{MODEL_NAME}.h5')
        print(f'Saved {MODEL_NAME} model.')
    except Exception as ex:
        print(f'ERROR: Failed to save model "{MODEL_NAME}". Reason="{ex}"')


def load_model(model_name: str):
    ''' Loads a saved model from JSON and H5PY '''
    try:
        json_file = open(f'{model_name}.json', 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights(f'{model_name}.h5')
        model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=METRICS)
        print(f'Loaded {model_name} model.')
        return model
    except Exception as ex:
        print(f'ERROR: Cannot open {model_name} file. Reason="{ex}"')


def bs():
    img_gt = image.load_img(
        'images/spectrograms/great_tit/great_tit_1865_0.jpg',
        target_size=(64, 64),
    )
    img_gt = image.img_to_array(img_gt)
    img_gt = np.expand_dims(img_gt, axis=0)

    img_yh = image.load_img(
        'images/spectrograms/yellowhammer/yellowhammer_1095_7.jpg',
        target_size=(64, 64),
    )
    img_yh = image.img_to_array(img_yh)
    img_yh = np.expand_dims(img_yh, axis=0)

    img_cb = image.load_img(
        'images/spectrograms/common_blackbird/common_blackbird_1_1.jpg',
        target_size=(64, 64),
    )
    img_cb = image.img_to_array(img_cb)
    img_cb = np.expand_dims(img_cb, axis=0)


def main():
    args = docopt(
        """
    Usage:
        cnn.py [options] <model_name>

    Options:
        -b, --batch-size NUM                Size of the batch
        -e, --epochs NUM                    No. of epochs
        -o, --optimizer STR                 Keras optimizer
        -l, --loss-func STR                 Keras loss function
        -c, --classes NUM                   No. of classes
        -i, --image-shape row,col,channels  e.g. (64, 64, 3)
    """
    )

    global MODEL_NAME
    global BATCH_SIZE
    global EPOCHS
    global OPTIMIZER
    global LOSS_FUNCTION
    global NUM_CLASSES
    global IMAGE_SHAPE

    MODEL_NAME = args['<model_name>']

    if args['--batch-size']:
        BATCH_SIZE = int(args['--batch-size'])

    if args['--epochs']:
        EPOCHS = int(args['--epochs'])

    if args['--optimizer']:
        OPTIMIZER = args['--optimizer']

    if args['--loss-func']:
        LOSS_FUNCTION = args['--loss-func']

    if args['--classes']:
        NUM_CLASSES = int(args['--classes'])
        print(f'Number of classes expected is {NUM_CLASSES}')

    if args['--image-shape']:
        try:
            img_shape = args['--image-shape'].split(',')
            if len(img_shape) != 3:
                raise ValueError('--image-shape must be a 3x1 vector')
            IMAGE_SHAPE = tuple(map(int, img_shape))
        except Exception as ex:
            print(f'ERROR: Invalid --image-shape. Reason="{ex}"')
    train_new_cnn()


if __name__ == "__main__":
    main()
