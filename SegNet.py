import os

import tensorflow as tf
import numpy as np
import random

from drawings_object import draw_plots_bayes, draw_plots_bayes_external
from layers_object import conv_layer, up_sampling, max_pool, initialization, \
    variable_with_weight_decay
from evaluation_object import cal_loss, normal_loss, per_class_acc, get_hist, print_hist_summary, train_op
from inputs_object import get_filename_list, dataset_inputs, get_all_test_data
from scipy import misc
import time

FLAGS = tf.app.flags.FLAGS

# Flags for directory paths needed at runtime
tf.app.flags.DEFINE_string('runtime_dir', 'logs/tensorboard', 'Directory where to write event logs and checkpoints.')
tf.app.flags.DEFINE_string('data_dir', '/var/local/data/skugele/COMP8150/project/combined',
                           'Directory where to store/read data sets.')

# Flags for logging
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')
tf.app.flags.DEFINE_integer('summary_frequency', 50, 'How often to save summary results (used for tensorboard).')
tf.app.flags.DEFINE_integer('checkpoint_frequency', 250, 'How often to save summary results.')
tf.app.flags.DEFINE_integer('validate_frequency', 250, 'How often to calculate validation loss/accuracy.')

# Flags for termination criteria
tf.app.flags.DEFINE_integer('n_epochs', 1000000, 'Max number of training epochs.')

# Flags for algorithm parameters
tf.app.flags.DEFINE_float('learning_rate', 0.0005, 'The learning rate (eta) to be used for neural networks.')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Mini-batch size for training.')

# Input dimensions
tf.app.flags.DEFINE_list('input_dims', [96, 72, 3], 'Dimensions of input images (width x height x channels).')
tf.app.flags.DEFINE_integer('n_classes', 4, 'The number of image classes contained in the input images.')


class SegNet:
    def __init__(self):
        # Number classes possible per pixel
        self.n_classes = FLAGS.n_classes

        # Paths to dataset (training/test/validation) summary files
        self.train_file = os.path.join(FLAGS.data_dir, 'train.txt')
        self.val_file = os.path.join(FLAGS.data_dir, 'validate.txt')
        self.test_file = os.path.join(FLAGS.data_dir, 'test.txt')

        self.input_w, self.input_h, self.input_c = FLAGS.input_dims

        self.images_tr, self.labels_tr = None, None
        self.images_val, self.labels_val = None, None

        # Create placeholders
        self.batch_size_pl = tf.placeholder(tf.int64, shape=[], name="batch_size")
        self.is_training_pl = tf.placeholder(tf.bool, name="is_training")
        self.with_dropout_pl = tf.placeholder(tf.bool, name="with_dropout")
        self.keep_prob_pl = tf.placeholder(tf.float32, shape=None, name="keep_rate")

        self.inputs_pl = tf.placeholder(tf.float32, [None, self.input_h, self.input_w, self.input_c])
        self.labels_pl = tf.placeholder(tf.int64, [None, self.input_h, self.input_w, 1])

        ##################
        # SegNet Encoder #
        ##################

        # SegNet includes Local Contrast Normalization - Substituted for Local Response Normalization
        self.norm1 = tf.nn.lrn(self.inputs_pl, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='norm1')

        # First set of convolution layers
        self.conv1_1 = conv_layer(self.norm1, "conv1_1", [3, 3, 3, 64], self.is_training_pl)
        self.conv1_2 = conv_layer(self.conv1_1, "conv1_2", [3, 3, 64, 64], self.is_training_pl)
        self.pool1, self.pool1_index, self.shape_1 = max_pool(self.conv1_2, 'pool1')

        # Second set of convolution layers
        self.conv2_1 = conv_layer(self.pool1, "conv2_1", [3, 3, 64, 128], self.is_training_pl)
        self.conv2_2 = conv_layer(self.conv2_1, "conv2_2", [3, 3, 128, 128], self.is_training_pl)
        self.pool2, self.pool2_index, self.shape_2 = max_pool(self.conv2_2, 'pool2')

        # Third set of convolution layers
        self.conv3_1 = conv_layer(self.pool2, "conv3_1", [3, 3, 128, 256], self.is_training_pl)
        self.conv3_2 = conv_layer(self.conv3_1, "conv3_2", [3, 3, 256, 256], self.is_training_pl)
        self.conv3_3 = conv_layer(self.conv3_2, "conv3_3", [3, 3, 256, 256], self.is_training_pl)
        self.pool3, self.pool3_index, self.shape_3 = max_pool(self.conv3_3, 'pool3')

        # Fourth set of convolution layers
        self.conv4_1 = conv_layer(self.pool3, "conv4_1", [3, 3, 256, 512], self.is_training_pl)
        self.conv4_2 = conv_layer(self.conv4_1, "conv4_2", [3, 3, 512, 512], self.is_training_pl)
        self.conv4_3 = conv_layer(self.conv4_2, "conv4_3", [3, 3, 512, 512], self.is_training_pl)
        self.pool4, self.pool4_index, self.shape_4 = max_pool(self.conv4_3, 'pool4')

        # Fifth set of convolution layers
        self.conv5_1 = conv_layer(self.pool4, "conv5_1", [3, 3, 512, 512], self.is_training_pl)
        self.conv5_2 = conv_layer(self.conv5_1, "conv5_2", [3, 3, 512, 512], self.is_training_pl)
        self.conv5_3 = conv_layer(self.conv5_2, "conv5_3", [3, 3, 512, 512], self.is_training_pl)
        self.pool5, self.pool5_index, self.shape_5 = max_pool(self.conv5_3, 'pool5')

        ##################
        # SegNet Encoder #
        ##################

        # First set of deconvolution layers
        self.deconv5_1 = up_sampling(self.pool5, self.pool5_index, self.shape_5, self.batch_size_pl,
                                     name="unpool_5")
        self.deconv5_2 = conv_layer(self.deconv5_1, "deconv5_2", [3, 3, 512, 512], self.is_training_pl)
        self.deconv5_3 = conv_layer(self.deconv5_2, "deconv5_3", [3, 3, 512, 512], self.is_training_pl)
        self.deconv5_4 = conv_layer(self.deconv5_3, "deconv5_4", [3, 3, 512, 512], self.is_training_pl)

        # Second set of deconvolution layers
        self.deconv4_1 = up_sampling(self.deconv5_4, self.pool4_index, self.shape_4, self.batch_size_pl,
                                     name="unpool_4")
        self.deconv4_2 = conv_layer(self.deconv4_1, "deconv4_2", [3, 3, 512, 512], self.is_training_pl)
        self.deconv4_3 = conv_layer(self.deconv4_2, "deconv4_3", [3, 3, 512, 512], self.is_training_pl)
        self.deconv4_4 = conv_layer(self.deconv4_3, "deconv4_4", [3, 3, 512, 256], self.is_training_pl)

        # Third set of deconvolution layers
        self.deconv3_1 = up_sampling(self.deconv4_4, self.pool3_index, self.shape_3, self.batch_size_pl,
                                     name="unpool_3")
        self.deconv3_2 = conv_layer(self.deconv3_1, "deconv3_2", [3, 3, 256, 256], self.is_training_pl)
        self.deconv3_3 = conv_layer(self.deconv3_2, "deconv3_3", [3, 3, 256, 256], self.is_training_pl)
        self.deconv3_4 = conv_layer(self.deconv3_3, "deconv3_4", [3, 3, 256, 128], self.is_training_pl)

        # Fourth set of deconvolution layers
        self.deconv2_1 = up_sampling(self.deconv3_4, self.pool2_index, self.shape_2, self.batch_size_pl,
                                     name="unpool_2")
        self.deconv2_2 = conv_layer(self.deconv2_1, "deconv2_2", [3, 3, 128, 128], self.is_training_pl)
        self.deconv2_3 = conv_layer(self.deconv2_2, "deconv2_3", [3, 3, 128, 64], self.is_training_pl)

        # Fifth set of deconvolution layers
        self.deconv1_1 = up_sampling(self.deconv2_3, self.pool1_index, self.shape_1, self.batch_size_pl,
                                     name="unpool_1")
        self.deconv1_2 = conv_layer(self.deconv1_1, "deconv1_2", [3, 3, 64, 64], self.is_training_pl)
        self.deconv1_3 = conv_layer(self.deconv1_2, "deconv1_3", [3, 3, 64, 64], self.is_training_pl)

        with tf.variable_scope('conv_classifier') as scope:
            self.kernel = variable_with_weight_decay('weights', initializer=initialization(1, 64),
                                                     shape=[1, 1, 64, self.n_classes], wd=False)
            self.conv = tf.nn.conv2d(self.deconv1_3, self.kernel, [1, 1, 1, 1], padding='SAME')
            self.biases = variable_with_weight_decay('biases', tf.constant_initializer(0.0),
                                                     shape=[self.n_classes], wd=False)
            self.logits = tf.nn.bias_add(self.conv, self.biases, name=scope.name)

    def train(self):
        image_filename, label_filename = get_filename_list(self.train_file)
        val_image_filename, val_label_filename = get_filename_list(self.val_file)

        if self.images_tr is None:
            self.images_tr, self.labels_tr = dataset_inputs(image_filename, label_filename, FLAGS.batch_size,
                                                            self.input_w, self.input_h, self.input_c)
            self.images_val, self.labels_val = dataset_inputs(val_image_filename, val_label_filename,
                                                              FLAGS.batch_size, self.input_w, self.input_h,
                                                              self.input_c)

        loss, accuracy, predictions = cal_loss(logits=self.logits, labels=self.labels_pl, n_classes=self.n_classes)
        train, global_step = train_op(loss, FLAGS.learning_rate)

        tf.summary.scalar("global_step", global_step)
        tf.summary.scalar("total loss", loss)

        # Calculate total number of trainable parameters
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('Total Trainable Parameters: ', total_parameters)

        with tf.train.SingularMonitoredSession(
                # save/load model state
                checkpoint_dir=FLAGS.runtime_dir,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.n_epochs),
                       tf.train.CheckpointSaverHook(
                           checkpoint_dir=FLAGS.runtime_dir,
                           save_steps=FLAGS.checkpoint_frequency,
                           saver=tf.train.Saver()),
                       tf.train.SummarySaverHook(
                           save_steps=FLAGS.summary_frequency,
                           output_dir=FLAGS.runtime_dir,
                           scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()),
                       )],
                config=tf.ConfigProto(log_device_placement=True)) as mon_sess:

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=mon_sess)

            while not mon_sess.should_stop():

                image_batch, label_batch = mon_sess.raw_session().run([self.images_tr, self.labels_tr])
                feed_dict = {self.inputs_pl: image_batch,
                             self.labels_pl: label_batch,
                             self.is_training_pl: True,
                             self.keep_prob_pl: 0.5,
                             self.with_dropout_pl: True,
                             self.batch_size_pl: FLAGS.batch_size}

                step, _, training_loss, training_acc = mon_sess.run([global_step, train, loss, accuracy],
                                                                    feed_dict=feed_dict)

                print("Iteration {}: Train Loss{:9.6f}, Train Accu {:9.6f}".format(step, training_loss, training_acc))

                # Check against validation set
                if step % FLAGS.validate_frequency == 0:
                    sampled_losses = []
                    sampled_accuracies = []

                    hist = np.zeros((self.n_classes, self.n_classes))

                    for test_step in range(int(20)):
                        fetches_valid = [loss, accuracy, self.logits]
                        image_batch_val, label_batch_val = mon_sess.raw_session().run(
                            [self.images_val, self.labels_val])

                        feed_dict_valid = {self.inputs_pl: image_batch_val,
                                           self.labels_pl: label_batch_val,
                                           self.is_training_pl: True,
                                           self.keep_prob_pl: 1.0,
                                           self.with_dropout_pl: False,
                                           self.batch_size_pl: FLAGS.batch_size}

                        validate_loss, validate_acc, predictions = mon_sess.raw_session().run(fetches_valid,
                                                                                              feed_dict_valid)
                        sampled_losses.append(validate_loss)
                        sampled_accuracies.append(validate_acc)
                        hist += get_hist(predictions, label_batch_val)

                    print_hist_summary(hist)

                    # Average loss and accuracy over n samples from validation set
                    avg_loss = np.mean(sampled_losses)
                    avg_acc = np.mean(sampled_accuracies)

                    print("Iteration {}: Avg Val Loss {:9.6f}, Avg Val Acc {:9.6f}".format(step, avg_loss, avg_acc))

                coord.request_stop()
                coord.join(threads)

    def visual_results(self, dataset_type="TEST", indices=None, n_samples=3, model_file=None):

        with tf.Session() as sess:

            # Restore saved session
            saver = tf.train.Saver()

            if model_file is None:
                saver.restore(sess, tf.train.latest_checkpoint(FLAGS.runtime_dir))
            else:
                saver.restore(sess, os.path.join(FLAGS.runtime_dir, model_file))

            _, _, prediction = cal_loss(logits=self.logits, labels=self.labels_pl, n_classes=self.n_classes)

            test_type_path = None
            if dataset_type == 'TRAIN':
                test_type_path = self.train_file
            elif dataset_type == 'VAL':
                test_type_path = self.val_file
            elif dataset_type == 'TEST':
                test_type_path = self.test_file

            # Load images
            image_filenames, label_filenames = get_filename_list(test_type_path)
            images, labels = get_all_test_data(image_filenames, label_filenames)

            if not indices:
                indices = random.sample(range(len(images)), n_samples)

            # Keep images subset of length images_index
            images = [images[i] for i in indices]
            labels = [labels[i] for i in indices]

            pred_tot = []

            for image_batch, label_batch in zip(images, labels):
                image_batch = np.reshape(image_batch, [1, self.input_h, self.input_w, self.input_c])
                label_batch = np.reshape(label_batch, [1, self.input_h, self.input_w, 1])

                fetches = [prediction]
                feed_dict = {self.inputs_pl: image_batch,
                             self.labels_pl: label_batch,
                             self.is_training_pl: False,
                             self.keep_prob_pl: 0.5,
                             self.batch_size_pl: 1}
                pred = sess.run(fetches=fetches, feed_dict=feed_dict)
                pred = np.reshape(pred, [self.input_h, self.input_w])
                pred_tot.append(pred)

            draw_plots_bayes(images, labels, pred_tot)

    def visual_results_external_image(self, images, model_file):

        images = [misc.imresize(image, (self.input_h, self.input_w)) for image in images]

        with tf.Session() as sess:

            # Restore saved session
            saver = tf.train.Saver()

            if model_file is None:
                saver.restore(sess, tf.train.latest_checkpoint(FLAGS.runtime_dir))
            else:
                saver.restore(sess, os.path.join(FLAGS.runtime_dir, model_file))

            _, _, prediction = cal_loss(logits=self.logits,
                                        labels=self.labels_pl,
                                        n_classes=self.n_classes)
            prob = tf.nn.softmax(self.logits, dim=-1)

            pred_tot = []
            var_tot = []

            labels = []
            for i in range(len(images)):
                labels.append(np.array([[1 for x in range(self.input_w)] for y in range(self.input_h)]))

            inference_time = []
            start_time = time.time()

            for image_batch, label_batch in zip(images, labels):
                image_batch = np.reshape(image_batch, [1, self.input_h, self.input_w, self.input_c])
                label_batch = np.reshape(label_batch, [1, self.input_h, self.input_w, 1])

                fetches = [prediction]
                feed_dict = {self.inputs_pl: image_batch,
                             self.labels_pl: label_batch,
                             self.is_training_pl: False,
                             self.keep_prob_pl: 0.5,
                             self.batch_size_pl: 1}
                pred = sess.run(fetches=fetches, feed_dict=feed_dict)
                pred = np.reshape(pred, [self.input_h, self.input_w])

                pred_tot.append(pred)
                inference_time.append(time.time() - start_time)
                start_time = time.time()

            try:
                draw_plots_bayes_external(images, pred_tot)
                return pred_tot, var_tot, inference_time
            except:
                return pred_tot, var_tot, inference_time

    def predict(self, images, model_file=None):

        with tf.Session() as sess:
            # Restore saved session
            saver = tf.train.Saver()

            if model_file is None:
                saver.restore(sess, tf.train.latest_checkpoint(FLAGS.runtime_dir))
            else:
                saver.restore(sess, os.path.join(FLAGS.runtime_dir, model_file))

            predictions = tf.reshape(tf.argmax(tf.reshape(self.logits, [-1, self.n_classes]), -1),
                                     [len(images), self.input_h, self.input_w])

            image_batch = np.reshape(images, [len(images), self.input_h, self.input_w, self.input_c])

            feed_dict = {self.inputs_pl: image_batch,
                         self.is_training_pl: False,
                         self.with_dropout_pl: False,
                         self.batch_size_pl: len(images)}

            return sess.run(predictions, feed_dict=feed_dict)

    def test(self):
        image_filename, label_filename = get_filename_list(self.test_file)

        with tf.Session() as sess:
            # Restore saved session
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.runtime_dir))

            loss, accuracy, prediction = normal_loss(self.logits, self.labels_pl, self.n_classes)

            images, labels = get_all_test_data(image_filename, label_filename)

            NUM_SAMPLE = []
            for i in range(30):
                NUM_SAMPLE.append(2 * i + 1)

            acc_final = []
            iu_final = []
            iu_mean_final = []
            # uncomment the line below to only run for two times.
            # NUM_SAMPLE = [1, 30]
            NUM_SAMPLE = [1]
            for num_sample_generate in NUM_SAMPLE:

                loss_tot = []
                acc_tot = []

                hist = np.zeros((self.n_classes, self.n_classes))
                step = 0
                for image_batch, label_batch in zip(images, labels):
                    image_batch = np.reshape(image_batch, [1, self.input_h, self.input_w, self.input_c])
                    label_batch = np.reshape(label_batch, [1, self.input_h, self.input_w, 1])
                    # comment the code below to apply the dropout for all the samples
                    if num_sample_generate == 1:
                        feed_dict = {self.inputs_pl: image_batch, self.labels_pl: label_batch,
                                     self.is_training_pl: False,
                                     self.keep_prob_pl: 0.5, self.with_dropout_pl: False,
                                     self.batch_size_pl: 1}
                    else:
                        feed_dict = {self.inputs_pl: image_batch, self.labels_pl: label_batch,
                                     self.is_training_pl: False,
                                     self.keep_prob_pl: 0.5, self.with_dropout_pl: True,
                                     self.batch_size_pl: 1}

                    loss_per, acc_per, logit, pred = sess.run([loss, accuracy, self.logits, prediction],
                                                              feed_dict=feed_dict)

                    loss_tot.append(loss_per)
                    acc_tot.append(acc_per)
                    print("Image Index {}: TEST Loss{:6.3f}, TEST Accu {:6.3f}".format(step, loss_tot[-1],
                                                                                       acc_tot[-1]))
                    step = step + 1
                    per_class_acc(logit, label_batch, self.n_classes)
                    hist += get_hist(logit, label_batch)

                acc_tot = np.diag(hist).sum() / hist.sum()
                iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

                print("Total Accuracy for test image: ", acc_tot)
                print("Total MoI for test images: ", iu)
                print("mean MoI for test images: ", np.nanmean(iu))

                acc_final.append(acc_tot)
                iu_final.append(iu)
                iu_mean_final.append(np.nanmean(iu))

            return acc_final, iu_final, iu_mean_final


def main(argv=None):
    if not tf.gfile.Exists(FLAGS.runtime_dir):
        tf.gfile.MakeDirs(FLAGS.runtime_dir)

    segnet = SegNet()
    segnet.train()


if __name__ == '__main__':
    tf.app.run()
