# Dog Breed Classification

This project uses deep learning for dog breed classification.

The [VGG-16 model](https://arxiv.org/abs/1409.1556) is used to classify dog breeds.

This project has three components:

1. Web scrapping of dog images from google.
   See [here](https://github.com/oliveirampo/dog_breed_classification/blob/main/app/scr/download_image.py)
2. Use of deep learning to image classification.
3. Deployment of classifier to heroku.


### File Structure

* app/data/
    Input files and images required to train the classification model.
* app/model/
    Directory where model is stored.
* app/scr/
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

### Credits

Udacity Data Scientist Nanodegree for providing the guidelines of the project.
