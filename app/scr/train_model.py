import sys
import os

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model

from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np

from keras.layers import GlobalAvgPool2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

import download_image
import plot


def old_step():
    bottleneck_features = np.load('../data/bottleneck_features/vgg16.npz')
    print(bottleneck_features)
    # sys.exit('STOP')

    # Load pretrained VGG-16 model
    # bottleneck_features = np.load('../data/bottleneck_features/DogVGG16Data.npz')
    # train_vgg16 = bottleneck_features['train']
    # valid_vgg16 = bottleneck_features['valid']
    # test_vgg16 = bottleneck_features['test']

    # Define a model architecture (Model 1)
    # model = Sequential()
    # model.add(Flatten(input_shape=(7, 7, 512)))
    # model.add(Dense(n_categories, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # model.summary()

    # Define another model architecture (Model 2)
    # model = Sequential()
    # model.add(GlobalAvgPool2D(input_shape=(7, 7, 512)))
    # model.add(Dense(n_categories, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # model.summary()

    # Train the model
    # checkpointer = ModelCheckpoint(filepath='../data/checkpoint/dogvgg16.weights.best.hdf5',
    #                                verbose=1, save_best_only=True)

    # print(train_vgg16)
    # print(train_vgg16.shape)
    # print(train_targets.shape)
    # model.fit(train_vgg16, train_targets, epochs=20, validation_data=(valid_vgg16, valid_targets),
    #           callbacks=[checkpointer], verbose=1, shuffle=True)


def load_data(n_categories, output_directory):
    """Load text files with categories as subfolder names using Keras.

    :param n_categories: (int) Number of categories.
    :param output_directory: (str) Directory with files.
    :return:
        categories: (arr) List of categories.
        dog_files: (arr) List of files with name of images.
        dog_targets: (arr) List of targets in categorical shape.
    """

    data = load_files(output_directory)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), n_categories)
    categories = np.array(data['target_names'])
    return categories, dog_files, dog_targets


def decode_images(files):
    """Converts images to matrix.

    :param files: (arr) List of path if images.
    :return:
        data: (arr) Matrix with images in (num images, 224, 224, 3) format.
    """

    data = []

    for img_path in files:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        data.append(x)

    data = np.array(data) / 224
    return data


def classify_image_resnet50():
    """Example from Keras of how to classify single image with ResNet50."""

    model = ResNet50(weights='imagenet')

    img_path = '../data/dogs/train/affenpinscher/img_0.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3)[0])


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


def main():
    dog_breed_file_name = '../data/inp/dog_breed.txt'
    dog_breed_list = download_image.read_data(dog_breed_file_name)
    n_categories = len(dog_breed_list)

    n_images = 60
    train_directory = '../data/dogs/train'
    validation_directory = '../data/dogs/validation'
    test_directory = '../data/dogs/test'
    chromedriver = '../data/exe/chromedriver_linux64'

    # Download images.
    # download_image.download(dog_breed_list, n_images, train_directory, validation_directory, test_directory,
    #                         chromedriver)

    # Load images
    categories, train_files, train_targets = load_data(n_categories, train_directory)
    _, valid_files, valid_targets = load_data(n_categories, validation_directory)
    _, test_files, test_targets = load_data(n_categories, test_directory)

    # Print statistics about the dataset
    # print('There are {} total dog categories.'.format(len(categories)))
    # print('There are {} training dog images.'.format(len(train_files)))
    # print('There are {} validation dog images.'.format(len(valid_files)))
    # print('There are {} test dog images.'.format(len(test_files)))

    # Plot example images
    # train_targets_bool = train_targets == 1
    # fig = plot.plot_images(3, 4, train_files, train_targets_bool, categories)
    # fig.show()

    x_train = decode_images(train_files)
    x_valid = decode_images(valid_files)
    x_test = decode_images(test_files)

    # TODO - save features to file

    # classify_image_vgg16()
    # extract_features_from_an_arbitrary_intermediate_layer()
    # fine_tune_pretrained_model(x_train, train_targets, x_train, train_targets)
    # train_transfer_learning_vgg(x_train, train_targets, x_valid, valid_targets)


if __name__ == '__main__':
    main()
