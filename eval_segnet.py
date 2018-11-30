import collections
import random
import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix
from drawings_object import writeImage, display_image
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


def main(argv=None):
    n_examples = None
    indices = None
    # indices = [86, 87, 93, 99]
    indices, images, labels = load_data(TRAIN_MANIFEST, indices=indices, n=n_examples)
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
    stats_per_image_minus_background_only = filter(lambda stats: sum(map(lambda s: np.ceil(s),stats.f1_scores)) > 1, stats_per_image)
    stats_per_multi_object_images = filter(lambda stats: sum(map(lambda s: np.ceil(s),stats.f1_scores)) > 2, stats_per_image)

    best = sorted(stats_per_multi_object_images, key=lambda stats: stats.accuracy, reverse=True)[0:10]
    for stats in best:
        display_image(images[stats.index])
        display_image(writeImage(labels[stats.index], plot=False))
        display_image(writeImage(predictions[stats.index], plot=False))

    worst = sorted(stats_per_image_minus_background_only, key=lambda stats: stats.accuracy)[0:5]
    for stats in worst:
        display_image(images[stats.index])
        display_image(writeImage(labels[stats.index], plot=False))
        display_image(writeImage(predictions[stats.index], plot=False))




if __name__ == '__main__':
    tf.app.run()
