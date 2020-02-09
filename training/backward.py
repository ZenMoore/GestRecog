import tensorflow as tf
import numpy as np
import training.forward as forward
import xlrd
import os

NUM_EXAMPLES = 8 * 36
# correspondence: 0:rond, 1:right-croix, 2:left-croix, 3:foudre, 4:..., 5:..., 6:..., 7:...

BATCH_SIZE = 4
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
MOVING_AVERAGE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAIN_STEPS = 30000 #todo use evaluation percentage threshold

MODEL_SAVE_PATH = '../models/'
MODEL_NAME = 'gr_model.ckpt'

DATASET_PATH = '../dataset/new/'

# 将存储数据的.xls文件转为(6, 256)手势序列numpy矩阵
def convert_to_numpy(data_file):
    book = xlrd.open_workbook(DATASET_PATH+'{}'.format(data_file))
    # book = xlrd.open_workbook(data_file)

    table = book.sheet_by_index(0)  # 这里是引入第一个sheet，默认是引入第一个，要引入别的可以改那个数字

    # nrows=table.nrows
    # ncols=table.ncols

    start = 1
    end = 257  # 这两个数是为了避开标题行，手动避的
    # rows=start-end

    list_values = []
    for i in range(1, 7):
        values = []
        for x in range(start, end):
            row = table.row_values(x)

            values.append(row[i])
        list_values.append(values)
    # print(list_values)
    data = np.array(list_values)
    return data

# 从数据集路径中读取数据并转为两个numpy联合起来的数组作为数据集 [datas, labels]
def get_dataset(path):
    datas = np.empty(shape= [288, 6, 256])
    labels = np.empty(shape= [288])
    count_label = 0 # 访问到了第几个数据集子文件夹
    count_sequence = 0
    dirs = os.listdir(DATASET_PATH)
    for dir in dirs:
        # 赋值 36 个 labels
        labels[count_label * 36 : (count_label + 1) * 36] = np.ones(shape= [36]) * count_label
        count_label += 1
        # 赋值 36 个手势序列
        for file in os.listdir(DATASET_PATH+'{}'.format(dir)):
            datas[count_sequence] = convert_to_numpy(dir+'/'+file)
            count_sequence += 1
    return [datas, labels]

# todo 给定上一个batch的末位置加一作为所返回batch的首位置
def next_batch(dataset, pos):
    return np.ones(BATCH_SIZE), np.ones(BATCH_SIZE)

# todo
def train(dataset):
    x = tf.placeholder(dtype= tf. float32, shape= [None, forward.INPUT_NODE])
    return 0

def main(argv = None):
    dataset = get_dataset(DATASET_PATH)
    train(dataset)

if __name__ == '__main__':
    tf.app.run()


