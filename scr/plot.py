import matplotlib.pyplot as plt
import cv2


def plot_images(n_rows, n_cols, files, targets, categories):
    """Plot images.

    :param n_rows: (int) Number of rows.
    :param n_cols: (int) Number of columns.
    :param files: (np.arr) List with name of image files.
    :param categories: (dict)
    :param targets: (np.arr)
    :return fig: (Figure object)
    """

    categories_list = list(categories.values())
    print(categories_list)

    fig = plt.figure(figsize=(20, 10))
    for i in range(12):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, xticks=[], yticks=[])
        # visualize_img(files[i], categories_list[targets[i]][0], ax)

        # TODO - fix here
        import sys
        sys.exit(123)
        print(targets[i])
        print(categories_list[targets[i]])
        # print(categories_list[targets[i]][0])

    return fig


def visualize_img(img_path, text, ax):
    """Visualize image given path"""
    img = cv2.imread(img_path)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.text(0, -5, text, fontsize=12)
