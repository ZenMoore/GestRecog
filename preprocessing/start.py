import numpy as np

# 这两个数据是根据JY61上位机得到的数据给出的，而不是根据本项目上位机得到的数据
SIGNAL_AMP = 0.5
FILTER_SIZE = 5

# 给定一个.xls文件，将其转变为(6, 256)维 numpy 向量
def convert_to_numpy(data_file):
    return np.empty([6, 256], dtype = np.float)

# method: convolution
def noise_reduction(data):
    for i in range(6):
        X = data[i]
        wave = np.ones(FILTER_SIZE)*SIGNAL_AMP
        data[i] = np.convolve(X, wave, 'same')
    return data

def zero_mean(data):
    data -= np.mean(data, axis=0)
    return data

def normalization(data):
    data /= np.std(data, axis=0)
    return data

def whitening(data):
    data -= np.mean(data, axis=0)  # zero-center the data (important)
    cov = np.dot(data.T, data) / data.shape[0]  # get the data covariance matrix
    U, S, V = np.linalg.svd(cov)
    data = np.dot(data, U)  # decorrelate the data
    return data

def run(data_file):
    data = convert_to_numpy(data_file)
    data = noise_reduction(data)
    data = zero_mean(data)
    data = normalization(data)
    data = whitening(data)
    return data