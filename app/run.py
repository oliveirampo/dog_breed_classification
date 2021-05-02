import base64

from flask_bootstrap import Bootstrap
from flask_flatpages import FlatPages
from flask import render_template
from flask_frozen import Freezer
from flask import redirect
from flask import request
from flask import flash
from flask import Flask
from flask import make_response
from flask import jsonify
from werkzeug.utils import secure_filename
import json
import sys
import os
from io import BytesIO
from PIL import Image


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

@app.route('/', methods=['GET', 'POST'])
def page_index():

    if request.method == 'POST':
        data_json = request.get_data()

        if data_json:
            if type(data_json) == bytes:
                data_json = data_json.decode('utf8')

            img_data = data_json.split(',')[1]

            f = open('/home/marina/lixo/lixo.png', 'wb')
            f.write(base64.b64decode(img_data))
            f.close()

            prediction = getPrediction('image')

            return jsonify(prediction)

    return render_template('index.html')


def getPrediction(image):
    return [{"breed": "dog1", "percentage": "(0.0 %)"},
            {"breed": "dog1", "percentage": "(0.0 %)"},
            {"breed": "dog3", "percentage": "(0.0 %)"}]


if __name__ == '__main__':
    app.run(debug=True)
