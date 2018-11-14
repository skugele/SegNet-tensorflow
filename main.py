import tensorflow as tf

from SegNet import SegNet

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('runtime_dir', 'logs/tensorboard', 'Directory where to write event logs and checkpoints.')
tf.app.flags.DEFINE_string('data_dir', 'SegNet/CamVid', 'Directory where to store/read data sets.')

# tf.app.flags.DEFINE_string('images_prefix', 'image_plot', 'The prefix to add to sampled images files.')
# tf.app.flags.DEFINE_string('images_file_ext', 'png', 'The file extension to use for sampled images files.')

# Flags for logging
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')
tf.app.flags.DEFINE_integer('log_frequency', 50, 'How often to log results to the console.')

# Flags for termination criteria
tf.app.flags.DEFINE_integer('n_epochs', 10000, 'Max number of training epochs.')

# Flags for algorithm parameters
tf.app.flags.DEFINE_float('learning_rate', 0.0003, 'The learning rate (eta) to be used for neural networks.')
tf.app.flags.DEFINE_integer('batch_size', 1, 'Mini-batch size from training.')

# Input dimensions
tf.app.flags.DEFINE_list('input_dims', [480, 360, 3], 'Dimensions of input images (width x height x channels).')
tf.app.flags.DEFINE_integer('n_classes', 12, 'The number of image classes contained in the input images.')


# {
#   "TRAIN_FILE": "SegNet/CamVid/train.txt",
#   "VAL_FILE": "SegNet/CamVid/val.txt",
#   "TEST_FILE": "SegNet/CamVid/test.txt",
#   "IMG_PREFIX": ".",
#   "LABEL_PREFIX": ".",
#   "BAYES": false,
#   "OPT": "ADAM",
#   "SAVE_MODEL_DIR": "./result/training/model",
#   "INPUT_HEIGHT": 360,
#   "INPUT_WIDTH": 480,
#   "INPUT_CHANNELS": 3,
#   "NUM_CLASSES": 12,
#   "USE_VGG": false,
#   "VGG_FILE": "vgg16.npy",
#   "TB_LOGS": "logs/tensorboard",
#   "BATCH_SIZE": 1
# }


def main(argv=None):
    if not tf.gfile.Exists(FLAGS.runtime_dir):
        tf.gfile.MakeDirs(FLAGS.runtime_dir)

    segnet = SegNet(FLAGS)
    segnet.train()


if __name__ == '__main__':
    tf.app.run()
