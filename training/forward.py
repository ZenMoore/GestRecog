import tensorflow as tf
slim = tf.contrib.slim

INPUT_NODE_HORIZONTAL = 6
INPUT_NODE_VERTICAL = 256
OUTPUT_NODE = 8

# input_tensor: (?, 6, 256)
# output: (?,8)
def inference(input_tensor, regularizer):
    return [0,0]