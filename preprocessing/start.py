import numpy as np
import preprocessing

# 给定一个.xls文件，将其转变为(6, 256)维 numpy 向量
def convert_to_numpy(data_file):
    return np.empty([6, 256], dtype = np.float)

def noise_reduction(data):
    return data

def zero_mean(data):
    data -= np.mean(data, axis=0)
    return data

def normalization(data):
    data /= np.std(data, axis=0)
    return data

def whitening(data):
    return data

def run(data_file):
    data = convert_to_numpy(data_file)
    data = noise_reduction(data)
    data = zero_mean(data)
    data = normalization(data)
    data = whitening(data)
    return data