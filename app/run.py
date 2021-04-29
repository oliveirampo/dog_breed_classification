from flask_bootstrap import Bootstrap
from flask_flatpages import FlatPages
from flask import render_template
from flask_frozen import Freezer
from flask import request
from flask import Flask
import json
import sys


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

@app.route('/')
def page_index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
