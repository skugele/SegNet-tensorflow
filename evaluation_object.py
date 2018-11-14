# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:12:45 2017
This file is utilized to calculate the loss, accuracy, per_class_accuracy, and MOI(mean of intersection)

"""
import tensorflow as tf
import numpy as np


def cal_loss(logits, labels, n_classes=12):
    loss_weight = np.array([
        0.2595,
        0.1826,
        4.5640,
        0.1417,
        0.9051,
        0.3826,
        9.6446,
        1.8418,
        0.6823,
        6.2478,
        7.3614,
        1.0974
    ])
    # class 0 to 11, but the class 11 is ignored, so maybe the class 11 is background!

    labels = tf.to_int64(labels)
    loss, accuracy, prediction = weighted_loss(logits, labels, number_class=n_classes, frequency=loss_weight)
    return loss, accuracy, prediction


def weighted_loss(logits, labels, number_class, frequency):
    """
    The reference paper is : https://arxiv.org/pdf/1411.4734.pdf 
    Median Frequency Balancing: alpha_c = median_freq/freq(c).
    median_freq is the median of these frequencies 
    freq(c) is the number of pixles of class c divided by the total number of pixels in images where c is present
    we weight each pixels by alpha_c
    Inputs: 
    logits is the output from the inference, which is the output of the decoder layers without softmax.
    labels: true label information 
    number_class: In the CamVid data set, it's 11 classes, or 12, because class 11 seems to be background? 
    frequency: is the frequency of each class
    Outputs:
    Loss
    Accuracy
    """
    label_flatten = tf.reshape(labels, [-1])
    label_onehot = tf.one_hot(label_flatten, depth=number_class)
    logits_reshape = tf.reshape(logits, [-1, number_class])
    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=label_onehot, logits=logits_reshape,
                                                             pos_weight=frequency)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.summary.scalar('loss', cross_entropy_mean)
    correct_prediction = tf.equal(tf.argmax(logits_reshape, -1), label_flatten)
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction))
    tf.summary.scalar('accuracy', accuracy)

    return cross_entropy_mean, accuracy, tf.argmax(logits_reshape, -1)


def normal_loss(logits, labels, number_class):
    """
    Calculate the normal loss instead of median frequency balancing
    Inputs:
    logits, the output from decoder layers, without softmax, shape [Num_batch,height,width,Number_class]
    lables: the atual label information, shape [Num_batch,height,width,1]
    number_class:12
    Output:loss,and accuracy
    Using tf.nn.sparse_softmax_cross_entropy_with_logits assume that each pixel have and only have one specific
    label, instead of having a probability belongs to labels. Also assume that logits is not softmax, because it
    will conduct a softmax internal to be efficient, this is the reason that we don't do softmax in the inference 
    function!
    """
    label_flatten = tf.to_int64(tf.reshape(labels, [-1]))
    logits_reshape = tf.reshape(logits, [-1, number_class])
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_flatten, logits=logits_reshape,
                                                                   name='normal_cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.summary.scalar('mean cross entropy', cross_entropy_mean)
    correct_prediction = tf.equal(tf.argmax(logits_reshape, -1), label_flatten)
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction))
    tf.summary.scalar('accuracy', accuracy)

    return cross_entropy_mean, accuracy, tf.argmax(logits_reshape, -1)


def per_class_acc(predictions, label_tensor, num_class):
    """
    This function is copied from "Implement slightly different segnet on tensorflow"
    """
    labels = label_tensor

    size = predictions.shape[0]
    hist = np.zeros((num_class, num_class))
    for i in range(size):
        hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    acc_total = np.diag(hist).sum() / hist.sum()
    print('accuracy = %f' % np.nanmean(acc_total))
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('mean IU  = %f' % np.nanmean(iu))
    for ii in range(num_class):
        if float(hist.sum(1)[ii]) == 0:
            acc = 0.0
        else:
            acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
        print("    class # %d accuracy = %f " % (ii, acc))


def fast_hist(a, b, n):
    """
    This function is copied from "Implement slightly different segnet on tensorflow"
    """
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def get_hist(predictions, labels):
    """
    This function is copied from "Implement slightly different segnet on tensorflow"
    """
    num_class = predictions.shape[3]  # becomes 2 for aerial - correct
    batch_size = predictions.shape[0]
    hist = np.zeros((num_class, num_class))
    for i in range(batch_size):
        hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    return hist


def print_hist_summary(hist):
    """
    This function is copied from "Implement slightly different segnet on tensorflow"
    """
    acc_total = np.diag(hist).sum() / hist.sum()
    print('accuracy = %f' % np.nanmean(acc_total))
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('mean IU  = %f' % np.nanmean(iu))
    for ii in range(hist.shape[0]):
        if float(hist.sum(1)[ii]) == 0:
            acc = 0.0
        else:
            acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
        print("    class # %d accuracy = %f " % (ii, acc))


def train_op(total_loss, learning_rate):
    global_step = tf.train.get_or_create_global_step()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        training_op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                             beta1=0.5).minimize(loss=total_loss,
                                                                 global_step=global_step,
                                                                 var_list=tf.trainable_variables())

    return training_op, global_step
