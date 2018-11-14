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


class SegNet:
    def __init__(self):
        self.n_classes = FLAGS.n_classes
        self.use_vgg = False

        if self.use_vgg is False:
            self.vgg_param_dict = None
            print("No VGG path in config, so learning from scratch")
        else:
            # self.vgg16_npy_path = self.config["VGG_FILE"]
            # self.vgg_param_dict = np.load(self.vgg16_npy_path, encoding='latin1').item()
            print("VGG parameter loaded")

        self.train_file = os.path.join(FLAGS.data_dir, 'train.txt')
        self.val_file = os.path.join(FLAGS.data_dir, 'val.txt')
        self.test_file = os.path.join(FLAGS.data_dir, 'test.txt')
        self.img_prefix = 'image_'
        self.label_prefix = 'label_'
        self.saved_dir = FLAGS.runtime_dir
        self.input_w, self.input_h, self.input_c = FLAGS.input_dims
        self.batch_size = FLAGS.batch_size
        self.n_epochs = FLAGS.n_epochs
        self.learning_rate = FLAGS.learning_rate

        self.train_loss, self.train_accuracy = [], []
        self.val_loss, self.val_acc = [], []

        self.model_version = 0  # used for saving the model
        self.saver = None
        self.images_tr, self.labels_tr = None, None
        self.images_val, self.labels_val = None, None

        # Create placeholders
        self.batch_size_pl = tf.placeholder(tf.int64, shape=[], name="batch_size")
        self.is_training_pl = tf.placeholder(tf.bool, name="is_training")
        self.with_dropout_pl = tf.placeholder(tf.bool, name="with_dropout")
        self.keep_prob_pl = tf.placeholder(tf.float32, shape=None, name="keep_rate")

        self.inputs_pl = tf.placeholder(tf.float32, [None, self.input_h, self.input_w, self.input_c])
        self.labels_pl = tf.placeholder(tf.int64, [None, self.input_h, self.input_w, 1])

        # Before enter the images into the architecture, we need to do Local Contrast Normalization
        # But it seems a bit complicated, so we use Local Response Normalization which implement in Tensorflow
        # Reference page:https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization
        self.norm1 = tf.nn.lrn(self.inputs_pl, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='norm1')
        # first box of convolution layer,each part we do convolution two times, so we have conv1_1, and conv1_2
        self.conv1_1 = conv_layer(self.norm1, "conv1_1", [3, 3, 3, 64], self.is_training_pl, self.use_vgg,
                                  self.vgg_param_dict)
        self.conv1_2 = conv_layer(self.conv1_1, "conv1_2", [3, 3, 64, 64], self.is_training_pl, self.use_vgg,
                                  self.vgg_param_dict)
        self.pool1, self.pool1_index, self.shape_1 = max_pool(self.conv1_2, 'pool1')

        # Second box of convolution layer(4)
        self.conv2_1 = conv_layer(self.pool1, "conv2_1", [3, 3, 64, 128], self.is_training_pl, self.use_vgg,
                                  self.vgg_param_dict)
        self.conv2_2 = conv_layer(self.conv2_1, "conv2_2", [3, 3, 128, 128], self.is_training_pl, self.use_vgg,
                                  self.vgg_param_dict)
        self.pool2, self.pool2_index, self.shape_2 = max_pool(self.conv2_2, 'pool2')

        # Third box of convolution layer(7)
        self.conv3_1 = conv_layer(self.pool2, "conv3_1", [3, 3, 128, 256], self.is_training_pl, self.use_vgg,
                                  self.vgg_param_dict)
        self.conv3_2 = conv_layer(self.conv3_1, "conv3_2", [3, 3, 256, 256], self.is_training_pl, self.use_vgg,
                                  self.vgg_param_dict)
        self.conv3_3 = conv_layer(self.conv3_2, "conv3_3", [3, 3, 256, 256], self.is_training_pl, self.use_vgg,
                                  self.vgg_param_dict)
        self.pool3, self.pool3_index, self.shape_3 = max_pool(self.conv3_3, 'pool3')

        # Fourth box of convolution layer(10)
        self.conv4_1 = conv_layer(self.pool3, "conv4_1", [3, 3, 256, 512], self.is_training_pl, self.use_vgg,
                                  self.vgg_param_dict)
        self.conv4_2 = conv_layer(self.conv4_1, "conv4_2", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                  self.vgg_param_dict)
        self.conv4_3 = conv_layer(self.conv4_2, "conv4_3", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                  self.vgg_param_dict)
        self.pool4, self.pool4_index, self.shape_4 = max_pool(self.conv4_3, 'pool4')

        # Fifth box of convolution layers(13)
        self.conv5_1 = conv_layer(self.pool4, "conv5_1", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                  self.vgg_param_dict)
        self.conv5_2 = conv_layer(self.conv5_1, "conv5_2", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                  self.vgg_param_dict)
        self.conv5_3 = conv_layer(self.conv5_2, "conv5_3", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                  self.vgg_param_dict)
        self.pool5, self.pool5_index, self.shape_5 = max_pool(self.conv5_3, 'pool5')

        # ---------------------So Now the encoder process has been Finished--------------------------------------#
        # ------------------Then Let's start Decoder Process-----------------------------------------------------#

        # First box of deconvolution layers(3)
        self.deconv5_1 = up_sampling(self.pool5, self.pool5_index, self.shape_5, self.batch_size_pl,
                                     name="unpool_5")
        self.deconv5_2 = conv_layer(self.deconv5_1, "deconv5_2", [3, 3, 512, 512], self.is_training_pl)
        self.deconv5_3 = conv_layer(self.deconv5_2, "deconv5_3", [3, 3, 512, 512], self.is_training_pl)
        self.deconv5_4 = conv_layer(self.deconv5_3, "deconv5_4", [3, 3, 512, 512], self.is_training_pl)
        # Second box of deconvolution layers(6)
        self.deconv4_1 = up_sampling(self.deconv5_4, self.pool4_index, self.shape_4, self.batch_size_pl,
                                     name="unpool_4")
        self.deconv4_2 = conv_layer(self.deconv4_1, "deconv4_2", [3, 3, 512, 512], self.is_training_pl)
        self.deconv4_3 = conv_layer(self.deconv4_2, "deconv4_3", [3, 3, 512, 512], self.is_training_pl)
        self.deconv4_4 = conv_layer(self.deconv4_3, "deconv4_4", [3, 3, 512, 256], self.is_training_pl)

        # Third box of deconvolution layers(9)
        self.deconv3_1 = up_sampling(self.deconv4_4, self.pool3_index, self.shape_3, self.batch_size_pl,
                                     name="unpool_3")
        self.deconv3_2 = conv_layer(self.deconv3_1, "deconv3_2", [3, 3, 256, 256], self.is_training_pl)
        self.deconv3_3 = conv_layer(self.deconv3_2, "deconv3_3", [3, 3, 256, 256], self.is_training_pl)
        self.deconv3_4 = conv_layer(self.deconv3_3, "deconv3_4", [3, 3, 256, 128], self.is_training_pl)
        # Fourth box of deconvolution layers(11)
        self.deconv2_1 = up_sampling(self.deconv3_4, self.pool2_index, self.shape_2, self.batch_size_pl,
                                     name="unpool_2")
        self.deconv2_2 = conv_layer(self.deconv2_1, "deconv2_2", [3, 3, 128, 128], self.is_training_pl)
        self.deconv2_3 = conv_layer(self.deconv2_2, "deconv2_3", [3, 3, 128, 64], self.is_training_pl)
        # Fifth box of deconvolution layers(13)
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
            self.images_tr, self.labels_tr = dataset_inputs(image_filename, label_filename, self.batch_size,
                                                            self.input_w, self.input_h, self.input_c)
            self.images_val, self.labels_val = dataset_inputs(val_image_filename, val_label_filename,
                                                              self.batch_size, self.input_w, self.input_h,
                                                              self.input_c)

        loss, accuracy, prediction = cal_loss(logits=self.logits, labels=self.labels_pl, n_classes=self.n_classes)
        train, global_step = train_op(loss, self.learning_rate)

        tf.summary.scalar("global_step", global_step)
        tf.summary.scalar("total loss", loss)

        with tf.train.SingularMonitoredSession(
                # save/load model state
                checkpoint_dir=self.saved_dir,
                hooks=[tf.train.StopAtStepHook(last_step=self.n_epochs),
                       # tf.train.NanTensorHook(self.train_loss),
                       tf.train.CheckpointSaverHook(
                           checkpoint_dir=self.saved_dir,
                           save_steps=1000,
                           saver=tf.train.Saver()),
                       tf.train.SummarySaverHook(
                           save_steps=100,
                           output_dir=self.saved_dir,
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
                             self.batch_size_pl: self.batch_size}

                step, _, _loss, _accuracy = mon_sess.run([global_step, train, loss, accuracy],
                                                         feed_dict=feed_dict)
                self.train_loss.append(_loss)
                self.train_accuracy.append(_accuracy)
                print("Iteration {}: Train Loss{:6.3f}, Train Accu {:6.3f}".format(step, self.train_loss[-1],
                                                                                   self.train_accuracy[-1]))

                if step % 100 == 0:
                    conv_classifier = mon_sess.run(self.logits, feed_dict=feed_dict)
                    print('per_class accuracy by logits in training time',
                          per_class_acc(conv_classifier, label_batch, self.n_classes))

                if step % 1000 == 0:
                    print("start validating.......")
                    _val_loss = []
                    _val_acc = []
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
                                           self.batch_size_pl: self.batch_size}
                        # since we still using mini-batch, so in the batch norm we set phase_train to be
                        # true, and because we didin't run the trainop process, so it will not update
                        # the weight!
                        _loss, _acc, _val_pred = mon_sess.raw_session().run(fetches_valid, feed_dict_valid)
                        _val_loss.append(_loss)
                        _val_acc.append(_acc)
                        hist += get_hist(_val_pred, label_batch_val)

                    print_hist_summary(hist)

                    self.val_loss.append(np.mean(_val_loss))
                    self.val_acc.append(np.mean(_val_acc))

                    print(
                        "Iteration {}: Train Loss {:6.3f}, Train Acc {:6.3f}, Val Loss {:6.3f}, Val Acc {:6.3f}".format(
                            step, self.train_loss[-1], self.train_accuracy[-1], self.val_loss[-1],
                            self.val_acc[-1]))

                coord.request_stop()
                coord.join(threads)

    def visual_results(self, dataset_type="TEST", images_index=3):

        with tf.Session() as sess:

            # Restore saved session
            saver = tf.train.Saver()
            chkpt = tf.train.latest_checkpoint(self.saved_dir)
            saver.restore(sess, chkpt)

            _, _, prediction = cal_loss(logits=self.logits, labels=self.labels_pl, n_classes=self.n_classes)
            prob = tf.nn.softmax(self.logits, dim=-1)

            test_type_path = None
            indexes = []
            if (dataset_type == 'TRAIN'):
                test_type_path = self.train_file
                if type(images_index) == list:
                    indexes = images_index
                else:
                    indexes = random.sample(range(367), images_index)
                # indexes = [0,75,150,225,300]
            elif (dataset_type == 'VAL'):
                test_type_path = self.val_file
                if type(images_index) == list:
                    indexes = images_index
                else:
                    indexes = random.sample(range(101), images_index)
                # indexes = [0,25,50,75,100]
            elif (dataset_type == 'TEST'):
                test_type_path = self.test_file
                if type(images_index) == list:
                    indexes = images_index
                else:
                    indexes = random.sample(range(233), images_index)

            # Load images
            image_filenames, label_filenames = get_filename_list(test_type_path)
            images, labels = get_all_test_data(image_filenames, label_filenames)

            # Keep images subset of length images_index
            images = [images[i] for i in indexes]
            labels = [labels[i] for i in indexes]

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

    def visual_results_external_image(self, images):

        images = [misc.imresize(image, (self.input_h, self.input_w)) for image in images]

        with tf.Session() as sess:

            # Restore saved session
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(self.saved_dir))

            _, _, prediction = cal_loss(logits=self.logits,
                                        labels=self.labels_pl)
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
                var_one = []

                pred_tot.append(pred)
                var_tot.append(var_one)
                inference_time.append(time.time() - start_time)
                start_time = time.time()

            try:
                draw_plots_bayes_external(images, pred_tot, var_tot)
                return pred_tot, var_tot, inference_time
            except:
                return pred_tot, var_tot, inference_time

    # def test(self, FLAGS):
    #     image_filename, label_filename = get_filename_list(self.test_file, FLAGS)
    #
    #     with self.graph.as_default():
    #         with self.sess as sess:
    #             loss, accuracy, prediction = normal_loss(self.logits, self.labels_pl, self.n_classes)
    #             prob = tf.nn.softmax(self.logits, dim=-1)
    #             prob = tf.reshape(prob, [self.input_h, self.input_w, self.n_classes])
    #
    #             images, labels = get_all_test_data(image_filename, label_filename)
    #
    #             NUM_SAMPLE = []
    #             for i in range(30):
    #                 NUM_SAMPLE.append(2 * i + 1)
    #
    #             acc_final = []
    #             iu_final = []
    #             iu_mean_final = []
    #             # uncomment the line below to only run for two times.
    #             # NUM_SAMPLE = [1, 30]
    #             NUM_SAMPLE = [1]
    #             for num_sample_generate in NUM_SAMPLE:
    #
    #                 loss_tot = []
    #                 acc_tot = []
    #                 pred_tot = []
    #                 var_tot = []
    #                 hist = np.zeros((self.n_classes, self.n_classes))
    #                 step = 0
    #                 for image_batch, label_batch in zip(images, labels):
    #                     image_batch = np.reshape(image_batch, [1, self.input_h, self.input_w, self.input_c])
    #                     label_batch = np.reshape(label_batch, [1, self.input_h, self.input_w, 1])
    #                     # comment the code below to apply the dropout for all the samples
    #                     if num_sample_generate == 1:
    #                         feed_dict = {self.inputs_pl: image_batch, self.labels_pl: label_batch,
    #                                      self.is_training_pl: False,
    #                                      self.keep_prob_pl: 0.5, self.with_dropout_pl: False,
    #                                      self.batch_size_pl: 1}
    #                     else:
    #                         feed_dict = {self.inputs_pl: image_batch, self.labels_pl: label_batch,
    #                                      self.is_training_pl: False,
    #                                      self.keep_prob_pl: 0.5, self.with_dropout_pl: True,
    #                                      self.batch_size_pl: 1}
    #                     # uncomment this code below to run the dropout for all the samples
    #                     # feed_dict = {test_data_tensor: image_batch, test_label_tensor:label_batch, phase_train: False, keep_prob:0.5, phase_train_dropout:True}
    #                     fetches = [loss, accuracy, self.logits, prediction]
    #                     loss_per, acc_per, logit, pred = sess.run(fetches=fetches, feed_dict=feed_dict)
    #                     var_one = []
    #
    #                     loss_tot.append(loss_per)
    #                     acc_tot.append(acc_per)
    #                     pred_tot.append(pred)
    #                     var_tot.append(var_one)
    #                     print("Image Index {}: TEST Loss{:6.3f}, TEST Accu {:6.3f}".format(step, loss_tot[-1],
    #                                                                                        acc_tot[-1]))
    #                     step = step + 1
    #                     per_class_acc(logit, label_batch, self.n_classes)
    #                     hist += get_hist(logit, label_batch)
    #
    #                 acc_tot = np.diag(hist).sum() / hist.sum()
    #                 iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    #
    #                 print("Total Accuracy for test image: ", acc_tot)
    #                 print("Total MoI for test images: ", iu)
    #                 print("mean MoI for test images: ", np.nanmean(iu))
    #
    #                 acc_final.append(acc_tot)
    #                 iu_final.append(iu)
    #                 iu_mean_final.append(np.nanmean(iu))
    #
    #         return acc_final, iu_final, iu_mean_final, pred_tot, var_tot


def main(argv=None):
    if not tf.gfile.Exists(FLAGS.runtime_dir):
        tf.gfile.MakeDirs(FLAGS.runtime_dir)

    segnet = SegNet()
    segnet.train()


if __name__ == '__main__':
    tf.app.run()
