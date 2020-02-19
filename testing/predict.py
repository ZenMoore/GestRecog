import os
import xlrd
import numpy as np
import application.inference as inf

# 本预测函数目前仅仅预测在训练集上面的正确率
# 使用的model为训练后供作测试的诸多models
# 而不是最佳model, 最佳model存储在application/model，这个只有出现更优模型时才会被更改！

DATASET_PATH = '../dataset/new/'
SEQUENCE_LEN = 256
STEP = 14001

type_map = {'right_to_left': 0,
            'round': 1,
            'croix': 2,
            'down_to_up': 3,
            'thunder': 4,
            'infinity': 5,
            'triangle': 6,
            'turn' : 7}


# 将存储数据的.xlsx文件转为(6, 256)手势序列numpy矩阵
def convert_to_numpy(data_file):
    book = xlrd.open_workbook(DATASET_PATH+'{}'.format(data_file))
    # book = xlrd.open_workbook(data_file)

    table = book.sheet_by_index(0)  # 这里是引入第一个sheet，默认是引入第一个，要引入别的可以改那个数字

    # nrows=table.nrows
    # ncols=table.ncols

    start = 0
    end = SEQUENCE_LEN + start  # 这两个数是为了避开标题行，手动避的
    # rows=start-end

    list_values = []
    for i in range(6): # 列
        values = []
        try:
            for x in range(start, end): #行

                row = table.row_values(x)

                values.append(row[i])
        except IndexError:
            # 捕捉混乱表格引起的错误
            print('Error in original: ' + data_file)
        list_values.append(values)
    # print(list_values)
    data = np.array(list_values)
    return data

# 返回全部 datas = [](一个个[6, 256]) 和 labels = [](一个个整数)
def get_info():
    dirs = os.listdir(DATASET_PATH)
    labels = []
    datas = []
    for dir in dirs:
        label = type_map.get(dir)
        print(label)
        for file in os.listdir(DATASET_PATH+'{}'.format(dir)):
            data = convert_to_numpy(dir + '/' + file)
            datas.append(data)
            labels.append(label)
    return datas, labels


if __name__ == "__main__":
    model_path = "../models/gr_model.ckpt" + '-' + str(STEP)
    datas, labels = get_info()
    count = 0
    print(len(labels))
    for i in range(len(labels)):
        out = inf.run(datas[i], model_path)
        print(out)
        print(labels[i])
        if out == labels[i]:
            count += 1
    print('After %d steps training, the precision in the whole dataset is %g.' %(STEP, count/len(labels)))