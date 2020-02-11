import tensorflow as tf
import numpy as np
import training.forward as forward
import xlrd
import os
import itertools

NUM_EXAMPLES = 8 * 36
# correspondence: 0:rond, 1:right-croix, 2:left-croix, 3:foudre, 4:..., 5:..., 6:..., 7:...

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
MOVING_AVERAGE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAIN_STEPS = 30000 #todo use evaluation percentage threshold

MODEL_SAVE_PATH = '../models/'
MODEL_NAME = 'gr_model.ckpt'

DATASET_PATH = '../dataset/new/'

SHUFFLE_BUFFER_SIZE = 4

batch_pos = 0

def convert_to_one_hot(tensor):
    a = tensor
    mydic = set()

    for i in range(288):
        item = a[i]
        mydic.add(item)
    # print(mydic)

    # 把这八个单个[]合起来,为了合并加了一个全是0的
    c = np.zeros(8)
    for item in mydic:
        b = np.zeros(8)
        b[int(item)] = 1
        c = np.vstack((c, b))
    # print(c)

    # 删去第一个全是0的
    index = [0]  # 最后一个的索引,因为最后一个是0
    c = np.array(list(itertools.compress(c, [i not in index for i in range(len(c))])))
    # print(c)

    # 每个重复36次
    e = np.zeros(8)
    for array in c:
        # print(array)
        d = np.tile(array, (36, 1))  # 重复36次
        e = np.vstack((e, d))
    # print(d)

    # 删去第一个空的
    index = [0]  # 最后一个的索引
    e = np.array(list(itertools.compress(e, [i not in index for i in range(len(e))])))
    # print(e)
    return e

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

# 返回Dataset对象，元素为字典项 ('data': (6, 256), 'label': 1)
def get_dataset(path):
    datas = np.zeros(shape= [NUM_EXAMPLES, 6, 256])
    labels = np.zeros(shape= [NUM_EXAMPLES], dtype= np.int32)
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
    labels = convert_to_one_hot(labels)
    dictionary = {'data':datas, 'label':labels} #todo 预感这个字典是可能弄乱数据的祸源
    # 经过测试，dataset 的每个元素为一个字典条目{'data':(6, 256), 'label':(8)}
    dataset = tf.data.Dataset.from_tensor_slices(dictionary)
    dataset.shuffle(SHUFFLE_BUFFER_SIZE) #todo 是一起打乱吗？
    return dataset

def train(dataset):
    # iterator = dataset.make_one_shot_iterator()

    data = tf.placeholder(dtype= tf.float32, shape= [None, forward.INPUT_NODE_HORIZONTAL, forward.INPUT_NODE_VERTICAL], name= 'data_input')
    label = tf.placeholder(dtype= tf.float32, shape= [None, forward.OUTPUT_NODE], name= 'label_input')

    regularizer = tf.contrib.layers.l1_regularizer(REGULARIZATION_RATE)# todo 可以修改l1范数正则为l2范数
    y = forward.inference(data, regularizer)
    global_step = tf.Variable(0, trainable= False)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits= y, labels= tf.argmax(label, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)# todo 可以尝试其他损失函数
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, NUM_EXAMPLES/BATCH_SIZE, LEARNING_RATE_DECAY)# todo 可以尝试更换衰减模型

    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)# todo 可以尝试更换滑动平均模型
    variable_average_op = variable_average.apply(tf.trainable_variables())

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step= global_step)# todo 可以尝试更换优化算法
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name= 'train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAIN_STEPS):
            data_feed = dataset.batch()['data'] # todo 怎么分离batch后的字典项
            label_feed = dataset.batch()['label']
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={data: data_feed, label: label_feed})

            if i%1000 == 0:
                print("After %d training step(s), loss on training batch is %g ." % (step, loss_value))
                # 给出global_step参数可以让每个被保存模型的文件名末尾加上训练的轮数，比如model.ckpt=1000表示训练1000论之后的模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv = None):
    dataset = get_dataset(DATASET_PATH)
    train(dataset)

if __name__ == '__main__':
    tf.app.run()


