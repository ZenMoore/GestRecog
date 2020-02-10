import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

INPUT_NODE_HORIZONTAL = 6
INPUT_NODE_VERTICAL = 256
OUTPUT_NODE = 8

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable('weights', shape= shape, initializer= tf.truncated_normal_initializer(stddev= 0.1)) # todo 可以尝试更换

    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))

    return weights

# convert (?, 6, 256) to (?, 6, 256, 1)
def add_one_dimension(input_tensor):
    output_tensor = np.zeros(shape=[2, 6, 256, 1])

    for i_batch in range(input_tensor.shape[0]):
        for i_width in range(input_tensor.shape[1]):
            for i_height in range(input_tensor.shape[2]):
                output_tensor[i_batch][i_width][i_height] = np.asarray(input_tensor[i_batch][i_width][i_height])
    return output_tensor

# input_tensor: (?, 6, 256)
# output: (?,8)
def inference(input_tensor, regularizer):
    biases = tf.get_variable('biases', [48], initializer=tf.constant_initializer(0.1))

    input_tensor = add_one_dimension(input_tensor)

    with tf.variable_scope('conv01'):
        filter_weight = tf.get_variable('filter_weight', [1, 3, 1, 12], initializer= tf.truncated_normal_initializer(stddev= 0.1)) # todo 可以尝试更换
        conv01 = tf.nn.conv2d(input_tensor, filter_weight, strides= [1, 1, 2, 1], padding='SAME')

    with tf.variable_scope('conv02'):
        filter_weight = tf.get_variable('filter_weight', [1, 3, 1, 24], initializer= tf.truncated_normal_initializer(stddev= 0.1)) # todo 可以尝试更换
        conv02 = tf.nn.conv2d(conv01, filter_weight, strides= [1, 1, 2, 1], padding='SAME')

    with tf.variable_scope('conv03'):
        filter_weight = tf.get_variable('filter_weight', [1, 3, 1, 48], initializer= tf.truncated_normal_initializer(stddev= 0.1)) # todo 可以尝试更换
        conv03 = tf.nn.conv2d(conv02, filter_weight, strides= [1, 1, 2, 1], padding='SAME')

    with tf.variable_scope('conv04'):
        filter_weight = tf.get_variable('filter_weight', [1, 3, 1, 48], initializer= tf.truncated_normal_initializer(stddev= 0.1)) # todo 可以尝试更换
        conv04 = tf.nn.conv2d(conv03, filter_weight, strides= [1, 1, 1, 1], padding='SAME')

    with tf.variable_scope('conv_to_recur'):
        # todo
        ...
    return [0,0]