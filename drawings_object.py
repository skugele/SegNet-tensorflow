import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.patches as mpatches

background = [50, 50, 50]
cylinder = [95, 200, 200]  # Cyan
cube = [150, 100, 200]  # Purple
sphere = [200, 50, 50]  # Pale Red


def display_image(image):
    plt.imshow(image)
    plt.show()
    plt.close()


def writeImage(image, plot=True):
    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array([background, cylinder, cube, sphere])
    for l in range(0, 4):
        r[image == l] = label_colours[l, 0]
        g[image == l] = label_colours[l, 1]
        b[image == l] = label_colours[l, 2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:, :, 0] = r / 1.0
    rgb[:, :, 1] = g / 1.0
    rgb[:, :, 2] = b / 1.0
    im = Image.fromarray(np.uint8(rgb))

    if plot:
        plt.imshow(im)
    else:
        return im


def display_color_legend():
    patches = [mpatches.Patch(color=np.asarray(background) / 256, label='background'),
               mpatches.Patch(color=np.asarray(cylinder) / 256, label='cylinder'),
               mpatches.Patch(color=np.asarray(cube) / 256, label='cube'),
               mpatches.Patch(color=np.asarray(sphere) / 256, label='sphere'), ]

    plt.figure(figsize=(0.2, 0.2))
    plt.legend(handles=patches, ncol=4)
    plt.axis('off')
    plt.show()


def draw_plots_bayes(images, labels, predicted_labels):
    rows = ['Image {}'.format(row) for row in range(1, len(images) + 1)]
    cols = ['Input', 'Ground truth', 'Output']

    nrows = len(rows)
    ncols = len(cols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 4))

    for i, row in enumerate(rows, start=0):

        plt.subplot(nrows, ncols, (ncols * i + 1))
        plt.imshow(images[i])
        # plt.ylabel(rows[i], size='22')
        plt.xticks([])
        plt.yticks([])

        if (i == 0):
            plt.title(cols[0], size='22', va='bottom')

        plt.subplot(nrows, ncols, (ncols * i + 2))
        writeImage(labels[i])
        plt.xticks([])
        plt.yticks([])

        if (i == 0):
            plt.title(cols[1], size='22', va='bottom')

        plt.subplot(nrows, ncols, (ncols * i + 3))
        writeImage(predicted_labels[i])
        plt.xticks([])
        plt.yticks([])

        if (i == 0):
            plt.title(cols[2], size='22', va='bottom')

    plt.show()


def draw_plots_bayes_external(images, predicted_labels):
    rows = ['Image {}'.format(row) for row in range(1, len(images) + 1)]
    cols = ['Input', 'Output']

    ncols = len(cols)
    nrows = len(rows)

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4))

    for i, row in enumerate(rows, start=0):

        plt.subplot(nrows, ncols, (ncols * i + 1))
        plt.imshow(images[i])
        plt.ylabel(row, size='18')
        plt.xticks([])
        plt.yticks([])

        if (i == 0):
            plt.title(cols[0], size='18', va='bottom')

        plt.subplot(nrows, ncols, (ncols * i + 2))
        writeImage(predicted_labels[i])
        plt.xticks([])
        plt.yticks([])

        if (i == 0):
            plt.title(cols[1], size='18', va='bottom')

    plt.show()
