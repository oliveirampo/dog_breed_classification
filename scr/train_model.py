import pandas as pd
import numpy as np
import json
import sys
import os

# from tensorflow.keras.applications.inception_v3 import InceptionV3
# from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
# from tensorflow.keras.models import Model
#
# from sklearn.datasets import load_files
# from keras.utils import np_utils
# import numpy as np
#
# from keras.layers import GlobalAvgPool2D
# from keras.layers import Dense
#
# from tensorflow.keras.models import load_model

from keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical

import download_image
import plot


def classify_image_vgg16():
    """Example from Keras of how to classify single image with ResNet50."""

    model = VGG16(weights='imagenet')

    img_path = '../data/dogs/train/affenpinscher/img_0.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    # print('Predicted:', decode_predictions(features, top=3)[0])
    print(decode_predictions(features, top=3))


def extract_features_from_an_arbitrary_intermediate_layer():
    """Example from Keras of how to extract features from an arbitrary intermediate layer."""

    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

    img_path = '../data/dogs/train/affenpinscher/img_0.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    block4_pool_features = model.predict(x)
    print(block4_pool_features)


def fine_tune_pretrained_model(x_train, y_train, x_validation, y_validation):
    """Example from Keras of how to fine tune pretrained model."""

    base_model = InceptionV3(weights='imagenet', include_top=False)
    # base_model = VGG16(weights='imagenet', include_top=False)
    # print(base_model.summary())

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAvgPool2D()(x)
    # let's add a fully-connect layer
    x = Dense(1024, activation='relu')(x)
    # add a logistic layer -- let's say we have 200 classes
    predictions = Dense(y_train.shape[1], activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    # print(model.summary())

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional layers of the model
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # train the model on the new fata for a few epochs
    model.fit(x_train, y_train, epochs=2, validation_data=(x_validation, y_validation), verbose=1, shuffle=True)
    # model.fit(train_vgg16, train_targets, epochs=20, validation_data=(valid_vgg16, valid_targets),
    #           callbacks=[checkpointer], verbose=1, shuffle=True)

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)
    sys.exit('STOP')

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from tensorflow.keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit(x_train, y_train, epochs=10, validation_data=(x_validation, y_validation), verbose=1, shuffle=True)


def train_transfer_learning_vgg(x_train, y_train, x_validation, y_validation):
    """Fine tune pretrained VGG model."""

    base_model = VGG16(weights='imagenet', include_top=False)
    # print(base_model.summary())

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAvgPool2D()(x)
    # let's add a fully-connect layer
    x = Dense(1024, activation='relu')(x)
    # add a logistic layer -- with correct number of classes
    predictions = Dense(y_train.shape[1], activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    # print(model.summary())

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional layers of the model
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # train the model on the new fata for a few epochs
    model.fit(x_train, y_train, epochs=2, validation_data=(x_validation, y_validation), verbose=1, shuffle=True)
    # model.fit(train_vgg16, train_targets, epochs=20, validation_data=(valid_vgg16, valid_targets),
    #           callbacks=[checkpointer], verbose=1, shuffle=True)

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from vgg16. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)
    sys.exit('STOP')

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from tensorflow.keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit(x_train, y_train, epochs=10, validation_data=(x_validation, y_validation), verbose=1, shuffle=True)


def print_data_info(categories, train_files, valid_files, test_files):
    """Print basic information about the dataset"""

    print('There are {} total dog categories.'.format(len(categories)))
    print('There are {} training dog images.'.format(len(train_files)))
    print('There are {} validation dog images.'.format(len(valid_files)))
    print('There are {} test dog images.'.format(len(test_files)))


def images_to_array(data_dir, df, image_size, class_to_num):
    image_names = df['id']
    image_labels = df['breed']
    data_size = len(image_names)
    # data_size = 70

    X = np.zeros([data_size, image_size[0], image_size[1], image_size[2]], dtype=np.uint8)
    y = np.zeros([data_size, 1], dtype=np.uint8)

    for i in range(data_size):
        img_name = image_names[i]
        img_dir = '{}/{}.png'.format(data_dir, img_name)
        img_pixels = load_img(img_dir, target_size=image_size)
        X[i] = img_pixels
        y[i] = class_to_num[image_labels[i]]

    y = to_categorical(y)

    ind = np.random.permutation(data_size)
    X = X[ind]
    y = y[ind]

    return X, y


def main():
    n_images = 60

    train_directory = 'data/dogs/train'
    dog_breed_file_name = 'data/inp/dog_breed.txt'
    chromedriver = 'data/exe/chromedriver_linux64'

    dog_breed_list = download_image.read_data(dog_breed_file_name)
    n_categories = len(dog_breed_list)

    # Download images
    is_to_download = False
    if is_to_download:
        download_image.download(dog_breed_list, n_images, train_directory, chromedriver)

    # Get categories
    df = pd.read_csv('data/dogs/labels.csv')
    dog_breeds = sorted(df['breed'].unique())
    n_classes = len(dog_breeds)

    # Converting classes to numbers
    class_to_num = dict(zip(dog_breeds, range(n_classes)))
    num_to_class = dict(zip(range(n_classes), dog_breeds))

    # Load and convert images to array
    img_size = (224, 224, 3)
    X, y = images_to_array(train_directory, df, img_size, class_to_num)

    # Plot a few examples
    fig = plot.plot_images(3, 4, 12, X, y, num_to_class)
    fig.show()


if __name__ == '__main__':
    main()
