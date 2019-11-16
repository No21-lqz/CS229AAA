import numpy as np
import matplotlib.pyplot as plt
import util
import pandas
import time
#import kersa
import LIQIAN
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
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