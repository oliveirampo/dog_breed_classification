import base64

from flask_bootstrap import Bootstrap
from flask_flatpages import FlatPages
from flask import render_template
from flask_frozen import Freezer
from flask import request
from flask import Flask
from flask import jsonify

from io import BytesIO
from PIL import Image
import cv2

import numpy as np
import json
import sys
import os

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


# Some configuration, ensures
# 1. Pages are loaded on request.
# 2. File name extension for pages is Markdown.
DEBUG = True
FLATPAGES_AUTO_RELOAD = DEBUG
FLATPAGES_EXTENSION = '.md'
FREEZER_RELATIVE_URLS = True
FREEZER_IGNORE_404_NOT_FOUND = True
FREEZER_DESTINATION = '../docs'


app = Flask(__name__)
app.config.from_object(__name__)
Bootstrap(app)


pages = FlatPages(app)
freezer = Freezer(app)

file_name = 'app/data/inp/categories.json'
with open(file_name) as file:
    categories = json.load(file)
# print(categories)

# model_path = 'app/model'
# model = load_model(model_path)
model = VGG16(weights='imagenet')
# print(model.summary())


@app.route('/', methods=['GET', 'POST'])
def page_index():
    if request.method == 'POST':
        data_json = request.get_data()

        if data_json:
            if type(data_json) == bytes:
                data_json = data_json.decode('utf8')

            img_data = data_json.split(',')[1]

            # save image to file
            # image_bytes = base64.b64decode(img_data)
            # f = open('/home/marina/lixo/lixo.png', 'wb')
            # f.write(base64.b64decode(image_bytes))
            # f.close()

            prediction = getPrediction(img_data, categories, model)

            return jsonify(prediction)

    return render_template('index.html')


def getPrediction(image_data, categories, model, top=3):
    """Generates output prediction for the input image."""

    img = Image.open(BytesIO(base64.b64decode(image_data)))
    img = img.resize((224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)

    if model.name == 'vgg16':
        results = decode_predictions(features, top=3)
        results = [{"breed": res[1].replace('_', ' '), 'percentage': '({:.2f} %)'.format(res[2] * 100.0)} for res in
                   results[0]]
        return results

    else:
        features = features[0]

    top_indices = features.argsort()[-top:][::-1]
    results = [{"breed": categories[str(i)].replace('_', ' '), "percentage": '({:.2f} %)'.format(features[i] * 100.0)}
               for i in top_indices]

    return results


if __name__ == '__main__':
    app.run(debug=True)
