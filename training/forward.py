import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

BATCH_SIZE = 4

INPUT_NODE_HORIZONTAL = 6
INPUT_NODE_VERTICAL = 256
OUTPUT_NODE = 8

GRU_HIDDEN_SIZE = 256
GRU_NUM_STEPS = 8

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable('weights', shape= shape, initializer= tf.truncated_normal_initializer(stddev= 0.1)) # todo 可以尝试更换

    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))

    return weights

# convert (?, 6, 256) to (?, 6, 256, 1)
def add_one_dimension(input_tensor):

    return tf.expand_dims(input_tensor, [-1])

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
        bn = tf.layers.batch_normalization(conv04, training= (regularizer != None))
        bias = tf.nn.bias_add(bn, biases) # dropout 和 bias 的顺序不知道是不是正确
        actived = tf.nn.tanh(bias)
        if (regularizer != None) : dropout = tf.nn.dropout(actived, 0.75)

    with tf.variable_scope('gru_layers'):

        # todo 是否需要 reshape 输入

        print(tf.shape(actived[:,0,:]))
        tf.reshape(actived[:,0,:], [BATCH_SIZE, -1])
        print(tf.shape(actived[:, 0, :]))

        # todo 可以使用 dropoutWrapper
        gru = tf.nn.rnn_cell.GRUCell(GRU_HIDDEN_SIZE, activation= tf.nn.relu)
        state = gru.zero_state(batch_size= BATCH_SIZE, dtype= tf.float32)
        # 使用static_rnn运行num_steps, 但是不能保证遍历全部时间点，可以更换为 dynamic_rnn
        # print(actived[:,0,:].shape)# todo debug
        for i in range(GRU_NUM_STEPS):
            if i > 0 : tf.get_variable_scope().reuse_variables()
            gru_output, state = gru(actived[:,i,:], state= state) # todo input_shape?

    with tf.variable_scope('fully_connected'):
        fc_weight = tf.get_variable('fc_weight', [GRU_HIDDEN_SIZE, OUTPUT_NODE], initializer= tf.truncated_normal_initializer(stddev= 0.1))

        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc_weight))
        fc_bias = tf.get_variable('bias', [OUTPUT_NODE], initializer= tf.constant_initializer(0.1))

        logits = tf.matmul(gru_output, fc_weight) + fc_bias
        y = tf.nn.softmax(logits=logits)

    return y