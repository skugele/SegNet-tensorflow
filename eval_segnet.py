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

random.seed(12345)


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


def load_data(manifest, n=None):
    image_filenames, label_filenames = get_filename_list(manifest)

    indices = range(0, len(label_filenames)) if n is None else np.random.choice(range(0, len(label_filenames)), n)

    image_filenames = [image_filenames[i] for i in indices]
    label_filenames = [label_filenames[i] for i in indices]

    images, labels = get_all_test_data(image_filenames, label_filenames)

    return images, labels



def get_confusion_matrix(labels, predictions, n_classes):

    # Flatten matrices
    labels = np.reshape(labels, (-1))
    predictions = np.reshape(predictions, (-1))

    return confusion_matrix(labels, predictions, labels=range(n_classes))

def main(argv=None):
    images, labels = load_data(TEST_MANIFEST, 10)
    segnet = SegNet()

    predictions = segnet.predict(images)

    for image, label, prediction in zip(images,labels,predictions):
        n_correct = np.sum((prediction == label).astype(int))
        print('Overall Accuracy: {}'.format(np.float(n_correct) / np.float(label.size)))

        cm = get_confusion_matrix(label, prediction, segnet.n_classes)
        print(cm)

        display_image(image)
        display_image(writeImage(label, plot=False))
        display_image(writeImage(prediction, plot=False))



if __name__ == '__main__':
    tf.app.run()
