import collections
import random
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec

from sklearn.metrics import confusion_matrix
from drawings_object import writeImage, display_image, background, cylinder, cube, sphere
from inputs_object import get_filename_list, get_all_test_data
from SegNet import SegNet

BACKGROUND_CATEGORY_ID = 0
CYLINDER_CATEGORY_ID = 1
CUBE_CATEGORY_ID = 2
SPHERE_CATEGORY_ID = 3

CHECKPOINT_DIR = 'logs/tensorfboard'

TRAIN_MANIFEST = '/var/local/data/skugele/COMP8150/project/combined/train.txt'
TEST_MANIFEST = '/var/local/data/skugele/COMP8150/project/combined/test.txt'
VALIDATE_MANIFEST = '/var/local/data/skugele/COMP8150/project/combined/validate.txt'

np.random.seed(12345)


def parse_manifest(path):
    with open(path) as fd:
        images_filenames = []
        mask_filenames = []
        categories = []

        for line in fd:
            tokens = line.strip().split(" ")

            images_filenames.append(tokens[0])
            mask_filenames.append(tokens[1])
            categories.append(tokens[2])

    return images_filenames, mask_filenames, categories


def load_data(manifest, indices=None, n=None):
    image_filenames, label_filenames = get_filename_list(manifest)

    if indices is None:
        indices = range(0, len(label_filenames)) if n is None else np.random.choice(range(0, len(label_filenames)), n)

    image_filenames = [image_filenames[i] for i in indices]
    label_filenames = [label_filenames[i] for i in indices]

    images, labels = get_all_test_data(image_filenames, label_filenames)

    return indices, np.array(images), np.array(labels)


def get_confusion_matrix(labels, predictions, n_classes):
    # Flatten matrices
    labels = np.reshape(labels, (-1))
    predictions = np.reshape(predictions, (-1))

    return confusion_matrix(labels, predictions, labels=range(n_classes))


def get_precision_and_recall(confusion_matrix, category):
    n_categories = confusion_matrix.shape[0]

    tp = confusion_matrix[category][category].astype(float)

    indices = np.arange(0, n_categories) != category

    fp = np.sum(confusion_matrix[indices, category]).astype(float)
    fn = np.sum(confusion_matrix[category, indices]).astype(float)

    if tp == 0:
        return 0, 0, 0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2.0 * (recall * precision) / (recall + precision)

    return precision, recall, f1_score


def get_accuracy(confusion_matrix):
    total_correct = np.sum(np.diag(confusion_matrix)).astype(float)
    total_possible = np.sum(confusion_matrix).astype(float)

    return total_correct / total_possible


Statistics = collections.namedtuple('Statistics',
                                    ['index', 'cm', 'accuracy', 'precisions', 'recalls', 'f1_scores'])


def get_stats_per_image(indices, images, labels, predictions, n_classes):
    stats = []

    for index, image, label, prediction in zip(indices, images, labels, predictions):
        cm = get_confusion_matrix(label, prediction, n_classes)

        accuracy = get_accuracy(cm)
        precisions = []
        recalls = []
        f1_scores = []

        for category in range(n_classes):
            precision, recall, f1_score = get_precision_and_recall(cm, category)

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)

        stats.append(Statistics(index, cm, accuracy, precisions, recalls, f1_scores))

    return stats


def get_diff(prediction, ground_truth):
    error_color = [200, 0, 0]
    diff_image = np.zeros((ground_truth.shape[0], ground_truth.shape[1], 3))
    diff_image[:, :] = [0, 200, 0]
    diff_image[prediction != ground_truth] = error_color

    return Image.fromarray(np.uint8(diff_image))


def display_results(plot_set):
    nrows = plot_set.size
    ncols = 4

    fig = plt.figure()

    spec = gridspec.GridSpec(nrows, ncols)
    spec.update(wspace=0.05, hspace=0.05)  # set the spacing between axes.

    for row in range(nrows):
        offset = row * ncols

        ax1 = fig.add_subplot(spec[row, 0])
        if row == 0:
            ax1.set_title('Image')

        imgplot = plt.imshow(plot_set.images[row])
        plt.xticks([])
        plt.yticks([])

        # plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

        ax2 = fig.add_subplot(spec[row, 1], sharex=ax1)
        if row == 0:
            ax2.set_title('Ground Truth')

        imgplot = plt.imshow(plot_set.ground_truths[row])
        plt.xticks([])
        plt.yticks([])

        ax3 = fig.add_subplot(spec[row, 2], sharex=ax1)
        if row == 0:
            ax3.set_title('Prediction')

        imgplot = plt.imshow(plot_set.predictions[row])
        plt.xticks([])
        plt.yticks([])

        ax4 = fig.add_subplot(spec[row, 3], sharex=ax1)
        if row == 0:
            ax4.set_title('Error Map')

        imgplot = plt.imshow(plot_set.error_map[row])
        plt.xticks([])
        plt.yticks([])

    # plt.subplots_adjust(wspace=0, hspace=0)

    # labels = ['background', 'cylinder', 'cube', 'sphere']
    # patches = [mpatches.Patch(color=np.asarray(background) / 256),
    #            mpatches.Patch(color=np.asarray(cylinder) / 256),
    #            mpatches.Patch(color=np.asarray(cube) / 256),
    #            mpatches.Patch(color=np.asarray(sphere) / 256), ]
    #
    # fig.legend(labels=labels, handles=patches, ncol=2, loc=2)

    plt.show()


def main(argv=None):
    n_examples = None
    indices = None
    # indices = [86, 87, 93, 99]
    indices, images, labels = load_data(VALIDATE_MANIFEST, indices=indices, n=n_examples)
    segnet = SegNet()

    predictions = segnet.predict(images, 'model.ckpt-4500')

    overall_cm = get_confusion_matrix(labels, predictions, segnet.n_classes)
    print('overall confusion matrix (all examples): \n', overall_cm)

    accuracy = get_accuracy(overall_cm)
    print('total accuracy (all categories): {}'.format(accuracy))

    for category in range(segnet.n_classes):
        precision, recall, f1_score = get_precision_and_recall(overall_cm, category)

        print('total precision (category = {}): {}'.format(category, precision))
        print('total recall (category = {}): {}'.format(category, recall))
        print('total f1_score (category = {}): {}'.format(category, f1_score))

    # Plot five "best" and five "worst" images + ground truth + predictions
    stats_per_image = get_stats_per_image(indices, images, labels, predictions, segnet.n_classes)

    # Filter out background images
    stats_per_image_minus_background_only = filter(lambda stats: sum(map(lambda s: np.ceil(s), stats.f1_scores)) > 1,
                                                   stats_per_image)
    stats_per_multi_object_images = filter(lambda stats: sum(map(lambda s: np.ceil(s), stats.f1_scores)) > 2,
                                           stats_per_image)

    PlotSet = collections.namedtuple('PlotSet', ['size', 'images', 'ground_truths', 'predictions', 'error_map'])

    best = sorted(stats_per_multi_object_images, key=lambda stats: stats.accuracy, reverse=True)[0:5]
    best_set = PlotSet(len(best), [], [], [], [])

    for index in map(lambda stats: stats.index, best):
        best_set.images.append(images[index])
        best_set.ground_truths.append(writeImage(labels[index], plot=False))
        best_set.predictions.append(writeImage(predictions[index], plot=False))
        best_set.error_map.append(get_diff(predictions[index], labels[index]))

    worst = sorted(stats_per_image_minus_background_only, key=lambda stats: stats.accuracy)[0:5]
    worst_set = PlotSet(len(worst), [], [], [], [])

    for index in map(lambda stats: stats.index, worst):
        worst_set.images.append(images[index])
        worst_set.ground_truths.append(writeImage(labels[index], plot=False))
        worst_set.predictions.append(writeImage(predictions[index], plot=False))
        worst_set.error_map.append(get_diff(predictions[index], labels[index]))

    display_results(best_set)
    display_results(worst_set)

    # images, ground_truths, predictions, diffs = [], [], [], []
    # worst = sorted(stats_per_image_minus_background_only, key=lambda stats: stats.accuracy)[0:5]
    #
    # worst_indices = map(lambda stats: stats.index, worst)
    #
    #
    # for stats in worst:
    #     display_image(images[stats.index])
    #     display_image(writeImage(labels[stats.index], plot=False))
    #     display_image(writeImage(predictions[stats.index], plot=False))
    #     display_image(get_diff(predictions[stats.index], labels[stats.index]))


if __name__ == '__main__':
    tf.app.run()
