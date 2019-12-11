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
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
import time
import datetime
from datetime import date
from matplotlib import pyplot as plt




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
    trend_day = np.zeros([len(trend_time), 1])
    publish_day = np.zeros([len(trend_time), 1])
    gap = np.zeros([len(trend_time), 1])
    for i in range(len(trend_time)):
        year = int(trend_time[i].split('/')[2])
        month = int(trend_time[i].split('/')[0])
        day = int(trend_time[i].split('/')[1])
        trend = date(year, month, day)
        year1 = int(publish_time[i][0:4])
        month1 = int(publish_time[i][5:7])
        day1 = int(publish_time[i][8:10])
        publish = date(year1, month1, day1)
        gap[i] = (trend - publish).days

        '''pt_year = int(publish_time[i][0:4])
        pt_month = int(publish_time[i][5:7])
        # tt_year = int('20' + trend_time[i][-2:])
        # tt_month = int(trend_time[i][6:8])
        tt_year = int(trend_time[i].split('/')[2])
        tt_month = int(trend_time[i].split('/')[0])
        time_gap.append(1 + (tt_year - pt_year) * 12 + (tt_month - pt_month)'''
    return gap

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
    """

    :param X_train:
    :param Y_train:
    :return:
    """
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
        print('ll: {:.6f}', ll)

    return theta'''

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

def predict(predict_p, theta):
    """
    theta : weights of 5 features
    :param y: n * 4 * 5 matrix needed to be labelled
    :return: return predicted label
    """
    n, m, z = predict_p.shape
    pro = np.zeros([n, m])
    for i in range(n):
        pro[i] = np.matmul(predict_p[i, :, :], theta).ravel()
    return pro

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
    return acc

def crossentropy(y_predict, y_label):
    n, m = y_predict.shape
    sum = 0
    for i in range(n):
        for j in range(m):
            if y_label[i, j] == 1:
                sum += -np.log(y_predict[i, j])
    return sum

def xgb_prediction(train, train_label, test, test_label):
    w_array = np.array([0.7] * train_label.shape[0])
    w_array[train_label == 0] = 0.9
    w_array[train_label == 1] = 8
    w_array[train_label == 3] = 1.7
    model = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=9,
                    min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                    objective= 'multi:softmax', num_class=4, nthread=4, scale_pos_weight=1, seed=27)
    eval_set = [(train, train_label), (test, test_label)]
    model.fit(train, train_label, eval_metric=["merror", "mlogloss"], eval_set=eval_set, sample_weight=w_array, verbose=True)
    results = model.evals_result()
    epochs = len(results['validation_0']['merror'])
    x_axis = range(0, epochs)

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(121)
    ax1.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
    ax1.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
    ax1.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Cross_entropy')
    plt.title('XGBoost Log Loss')

    ax2 = fig.add_subplot(122)
    ax2.plot(x_axis, results['validation_0']['merror'], label='Train')
    ax2.plot(x_axis, results['validation_1']['merror'], label='Test')
    ax2.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Prediction errors')
    plt.title('XGBoost predicted errors')
    plt.show()

    prediction = model.predict(test)
    return prediction

def xgb_prediction_mutli(train, train_label, test):
    model = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=9,
                    min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                    objective= 'binary:hinge', nthread=4, scale_pos_weight=1, seed=1)
    model.fit(train, train_label)
    prediction = model.predict(test)
    return prediction