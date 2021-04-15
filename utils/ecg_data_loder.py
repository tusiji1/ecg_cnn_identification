import wfdb
import os
import wfdb
import pywt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RATIO=0.3
# 小波去噪预处理
def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata

def getDataSet(data_path,number,i,data_set,lable_set):
    #print('读取'+number+'心电信号')
    record=wfdb.rdrecord(data_path+number,channel_names=['MLII'])
    data=record.p_signal.flatten()
    rdata=denoise(data)
    data_set.append(rdata[0:432000])
    #should be 0
    #print(number[1:4])
    #number=int(number[1:4])
    x=np.array([i]*864)
    list1=x.tolist()
    lable_set.extend(list1)

def load(data_path):
    # should flect to 0-n
    number_set=['100','101', '103', '105', '106', '107','108', '109', '111', '112', '113', '114'
                , '115','116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                 '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                 '231', '232', '233', '234']
    data_set=[]
    lable_set=[]
    i=0
    for n in number_set:
        getDataSet(data_path,n,i,data_set,lable_set)
        i+=1
    # 转numpy数组,打乱顺序
    print('读取完毕.........')
    data_set1 = np.array(data_set).reshape(-1,500)
    lable_set1 = np.array(lable_set).reshape(-1,1)
    train_ds = np.hstack((data_set1, lable_set1))

    np.random.shuffle(train_ds)
    X = train_ds[:, :500].reshape(-1, 500, 1)
    Y = train_ds[:, 500]
    shuffle_index = np.random.permutation(len(X))
    test_length = int(RATIO * len(shuffle_index))
    test_index = shuffle_index[:test_length]
    train_index = shuffle_index[test_length:]
    X_test, Y_test = X[test_index], Y[test_index]
    X_train, Y_train = X[train_index], Y[train_index]
    return X_train, Y_train, X_test, Y_test


