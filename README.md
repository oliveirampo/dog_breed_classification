# Dog Breed Classification (on construction)

This project uses deep learning for dog breed classification.

Transfer learning using pretrained models,
such as the
[VGG-16 model](https://arxiv.org/abs/1409.1556)
or the [ResNet50 model](https://arxiv.org/abs/1512.03385)
are used to classify dog breeds.

This project has three components:

1. Web scrapping of dog images from google.
   See [here](https://github.com/oliveirampo/dog_breed_classification/blob/main/scr/download_image.py).
2. Use of deep learning to image classification.
3. [Flask App](https://classify-me-auau.herokuapp.com/) deployed to heroku.


### File Structure

* data/
    Input files and images required to train the classification model.
* model/
    Directory where model is stored.
* scr/
    Directory for script files
* app/static/
    Directory for static files for API
* app/template/
    Directory with html file

There is also available a Collab
[Notebook](https://github.com/oliveirampo/dog_breed_classification/blob/main/app/scr/train_model.ipynb)
with preliminary steps
to perform transfer learning.
Note that this notebook is still on progress.

### Prerequisites

Python 3.8.8

The list of packages and recommended versions are listed
in the file `requirements.txt`

### Installation

Install the packages listed in `requirements.txt` preferably in a virtual environment.

```python
pip install -r requirements.txt
```

### Running

```python
python run.py
```

### Credits

Udacity Data Scientist Nanodegree for providing the guidelines of the project.
