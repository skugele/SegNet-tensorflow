"""
Created on Tue Nov 21 10:09:39 2017
This file is utilized to denote different layers, there are conv_layer, conv_layer_enc, max_pool, up_sampling
@author: s161488
"""
import numpy as np
import tensorflow as tf
import math


def max_pool(inputs, name):
    with tf.variable_scope(name) as scope:
        value, index = tf.nn.max_pool_with_argmax(tf.to_double(inputs), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                  padding='SAME', name=scope.name)
    return tf.to_float(value), index, inputs.get_shape().as_list()
    # here value is the max value, index is the corresponding index, the detail information is here
    # https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/nn/max_pool_with_argmax


def conv_layer(input_layer, name, shape, is_training):
    """
    Inputs:
    input_layer: The input image or tensor
    name: corresponding layer's name
    shape: the shape of kernel size
    is_training: represent if the weight should update
    """
    with tf.variable_scope(name) as scope:
        filter = variable_with_weight_decay('weights', initializer=initialization(shape[0], shape[2]),
                                          shape=shape, wd=False)
        tf.summary.histogram(scope.name + "weight", filter)
        conv = tf.nn.conv2d(input_layer, filter, [1, 1, 1, 1], padding='SAME')
        conv_biases = variable_with_weight_decay('biases', initializer=tf.constant_initializer(0.0),
                                                 shape=shape[3],
                                                 wd=False)
        tf.summary.histogram(scope.name + "bias", conv_biases)
        bias = tf.nn.bias_add(conv, conv_biases)
        conv_out = tf.nn.relu(batch_norm(bias, is_training, scope))
    return conv_out


def batch_norm(bias_input, is_training, scope):
    with tf.variable_scope(scope.name) as scope:
        return tf.cond(is_training,
                       lambda: tf.contrib.layers.batch_norm(bias_input, is_training=True, center=False, scope=scope),
                       lambda: tf.contrib.layers.batch_norm(bias_input, is_training=False, center=False, reuse=True,
                                                            scope=scope))


def up_sampling(pool, ind, output_shape, batch_size, name=None):
    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:   max pooled output tensor
           ind:      argmax indices
           ksize:     ksize is the same as for the pool
       Return:
           unpool:    unpooling tensor
           :param batch_size:
    """
    with tf.variable_scope(name):
        pool_ = tf.reshape(pool, [-1])
        batch_range = tf.reshape(tf.range(batch_size, dtype=ind.dtype), [tf.shape(pool)[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, [-1, 1])
        ind_ = tf.reshape(ind, [-1, 1])
        ind_ = tf.concat([b, ind_], 1)
        ret = tf.scatter_nd(ind_, pool_, shape=[batch_size, output_shape[1] * output_shape[2] * output_shape[3]])
        # the reason that we use tf.scatter_nd: if we use tf.sparse_tensor_to_dense, then the gradient is None, which will cut off the network.
        # But if we use tf.scatter_nd, the gradients for all the trainable variables will be tensors, instead of None.
        # The usage for tf.scatter_nd is that: create a new tensor by applying sparse UPDATES(which is the pooling value) to individual values of slices within a
        # zero tensor of given shape (FLAT_OUTPUT_SHAPE) according to the indices (ind_). If we ues the orignal code, the only thing we need to change is: changeing
        # from tf.sparse_tensor_to_dense(sparse_tensor) to tf.sparse_add(tf.zeros((output_sahpe)),sparse_tensor) which will give us the gradients!!!
        ret = tf.reshape(ret, [tf.shape(pool)[0], output_shape[1], output_shape[2], output_shape[3]])
        return ret


def initialization(k, c):
    """
    Here the reference paper is https:arxiv.org/pdf/1502.01852
    k is the filter size
    c is the number of input channels in the filter tensor
    we assume for all the layers, the Kernel Matrix follows a gaussian distribution N(0, \sqrt(2/nl)), where nl is 
    the total number of units in the input, k^2c, k is the spartial filter size and c is the number of input channels. 
    Output:
    The initialized weight
    """
    std = math.sqrt(2. / (k ** 2 * c))
    return tf.truncated_normal_initializer(stddev=std)


def variable_with_weight_decay(name, initializer, shape, wd):
    """
    Help to create an initialized variable with weight decay
    The variable is initialized with a truncated normal distribution, only the value for standard deviation is determined
    by the specific function _initialization
    Inputs: wd is utilized to notify if weight decay is utilized
    Return: variable tensor
    """
    var = tf.get_variable(name, shape, initializer=initializer)
    if wd is True:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    # tf.nn.l2_loss is utilized to compute half L2 norm of a tensor without the sqrt output = sum(t**2)/2
    return var
