# Dog Breed Classification

This project uses deep learning for dog breed classification.

Transfer learning using pretrained models,
such as
InceptionV3,
Xception,
InceptionResNetV2,
and
[VGG-16](https://arxiv.org/abs/1409.1556)
[models](https://keras.io/api/applications/)
are used to classify dog breeds.

This project has three components:

1. Web scrapping of dog images from google.
   See [here](https://github.com/oliveirampo/dog_breed_classification/blob/main/scr/download_image.py).
2. Use of deep learning to image classification.
3. [Flask App](https://classify-me-auau.herokuapp.com/) deployed to
   [Heroku](https://www.heroku.com).


### File Structure

* data/
    Input files and images required to train the classification model.
* model/
    Directory where model is stored.
* scr/
    Directory for script files
* flask_app/
    Directory for flask app

There is also available a Collab
[Notebook](https://github.com/oliveirampo/dog_breed_classification/blob/main/train_model.ipynb)
with steps
to perform transfer learning.

### Prerequisites

Python 3.8.8

The list of packages and recommended versions to run the notebook are listed
in the file `requirements.txt`.
Note that this notebook can also easily run in the Colab evironment,
but you need to authorize its access to your Drive.

The list of packages and recommended versions for the flask app
are listed in the file `flask_app/requirements.txt`

### Installation

Install the packages listed in `flas_app/requirements.txt` preferably in a virtual environment.

```python
pip install -r requirements.txt
```

### Running

The notebook can run in the Collab environment.

The flask app can run locally,
from the flask_app/ directory,
using the following command

```python
python run.py
```

Its online version is available
[here](https://classify-me-auau.herokuapp.com/). 
Note that, this works just as an example of how the model could be used,
as the app uses the ResNet50 model, without any modification (further training).
This choice was made due to memory constraints from Heroku,
which makes hard to deploy more accurate but also larger models.

### Credits

Udacity Data Scientist Nanodegree for providing the guidelines of the project,
and [khanrahim notebook](https://www.kaggle.com/khanrahim/dog-breed) as inspiration for the deep neural network architecture.
