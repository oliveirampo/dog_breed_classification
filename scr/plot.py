import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_images(num_rows, num_cols, num_axes, matrix, labels, categories):
    """Plot images given matrix.

    :param num_rows: (int) Number of rows in plot.
    :param num_cols: (int) Number of columns in plot.
    :param num_axes: (int) Number of axes to plot.
    :param matrix: (2D arr) Matrix with images.
    :param labels: (arr) Categorical data with possible classes.
    :param categories: (dict) Name of categories (dog breeds).
    :return fig: Figure object.
    """

    fig = plt.figure(figsize=(20, 10))

    for i in range(num_axes):
        ax = fig.add_subplot(num_rows, num_cols, i + 1, xticks=[], yticks=[])
        ax.imshow(matrix[i], interpolation='nearest')

        # Get index of item with value == 1. The result is an array of arrays.
        idx = str(np.where(labels[i] == 1)[0][0])
        breed = categories[idx]
        breed = breed.replace('_', ' ').title()

        ax.text(0, -5, breed, fontsize=14)

    return fig


def visualize_img(img_path, text, ax):
    """Visualize image given path"""
    img = cv2.imread(img_path)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.text(0, -5, text, fontsize=12)
