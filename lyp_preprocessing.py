import numpy as np
import matplotlib.pyplot as plt
import util
import pandas
import time
#import kersa
import LIQIAN
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import csv



def get_string_header(csvpath, header):
    """
    :param csvpath: name of csv flie (type: string, like '***.csv')
    :param header: input the header of column (type: string)
    :return: a list of string data you want (dim: n)
    """
    with open(csvpath, 'r', encoding='UTF-8') as csvfile:
        reader = csv.DictReader(csvfile)
        output = [row[header] for row in reader]
    return output

#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, )

def get_time_gap(publish_time, trend_time):
    """
    :param csvpath: name of csv flie (type: string, like '***.csv')
    :return: a list of time gap (dim: n)
    """
    time_gap = list()
    for i in range (len(publish_time)):
        pt_year = int(publish_time[i][0:4])
        pt_month = int(publish_time[i][5:7])
        tt_year = int('20' + trend_time[i][0:2])
        tt_month = int(trend_time[i][6:8])
        time_gap.append(1 + (tt_year - pt_year) * 12 + (tt_month - pt_month))
    return np.array(time_gap)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(theta, x):
    p = sigmoid(np.matmul(theta, x))
    return p

def create_3d(a, b, c, d, e):
    """
    :param a, b, c, d, e: different feature probabolity matrix (dim: n * 4)
    :return: 3D matrix (dim: n * 4 * 5)
    """
    n = a.shape[0]
    train_p = np.zeros([n, 4, 5])
    for i in range(n):
        train_p[i, :, :] = np.array([a[i], b[i], c[i], d[i], e[i]]).transpose()
    return train_p

def trainlogsitic(X_train, Y_train):
    n, m, z = X_train.shape
    theta = np.zeros([z, 1])
    new_x = np.zeros([n * 4, 5])
    new_x = X_train.reshape([n * 4, 5])
    new_y = Y_train.reshape([n * 4, 1]).ravel()
    print(new_x)
    print(new_y)
    model = LogisticRegression(max_iter=10000, solver='sag', tol=1e-6)
    result = model.fit(new_x, new_y)
    para = model.coef_
    print(para)
    return para.ravel()

def trainsoftmax(X_train, Y_train):
    '''
    itera = 0
    ll = 1
    pre_ll = 0
    while itera < max_iter and abs(ll - pre_ll) > 1e-5:
        sig= 0
        for i in range(z):
            sum = 0
            for j in range(n):
                for k in range(m):
                    if Y_train[i, k] == 1:
                        mul = np.matmul(X_train[j, k, :], theta)
                        sum += (-1) * (1 - sigmoid(mul)) * X_train[j, k, i] + theta[i]
            theta[i] += alpha * sum
        for i in range(n):
            mul = 0
            for j in range(m):
                if Y_train[i, j] == 1:
                    mul += np.matmul(X_train[i, j, :], theta)
            sig += (-1) * np.log(sigmoid(mul)) + 1/2 * lamda * np.linalg.norm(theta)
        sig = sig / n
        pre_ll = ll
        ll = sig
        itera += 1
        print('iteration: {:05d}', itera)
        print('ll: {:.6f}', ll)'''

    return theta

def normalization(target):
    """
    :param target: a n * 2 matrix with intercept
    :return: intercept is unchanged, and the variable in the other column is normalized (return n * 2 matrix)
    """
    max_t = max(target[:, 1])
    min_t = min(target[:, 1])
    print(max_t)
    print(min_t)
    n = target.shape[0]
    normal_target = np.zeros([n, 2])
    for i in range(n):
        normal_target[i, 0] = target[i, 0]
        normal_target[i, 1] = (target[i, 1] - min_t + 1) / (max_t - min_t)
    return normal_target

def predict_pro(predict_p, theta):
    n, m, z = predict_p.shape
    pro = np.zeros([n, 1])
    for i in range(n):
        pro[i] = np.matmul(predict_p[i, :, :], theta).argmax(axis=0)
    return pro

def predict(y):
    """

    :param y: n * 4 matrix needed to be labelled
    :return: return predicted label
    """
    n = y.shape[0]
    predict_label = np.zeros(n)
    predict_label = y.argmax(axis=1)
    return predict_label

def accuracy(y, y_predict):
    """

    :param : y is a original label matrix, y_predict is a predicted label matrix (n * 1)
    :return: accuracy
    """
    n = y.shape[0]
    count = 0
    for i in range(n):
        if y[i] == y_predict[i]:
            count += 1
    acc = count / n
    return

def crossentropy(y_predict, y_label):
    n, m = y_predict.shape
    sum = 0
    for i in range(n):
        for j in range(m):
            if y_label[i, j] == 1:
                sum += -np.log(y_predict[i, j])
    return sum