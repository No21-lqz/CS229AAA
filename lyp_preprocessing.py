import numpy as np
import matplotlib.pyplot as plt
import util
#import ntlk
import time
import keras

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

def get_time_gap(csvpath):
    """
    :param csvpath: name of csv flie (type: string, like '***.csv')
    :return: a list of time gap (dim: n)
    """
    publish_time = np.array(get_string_header(csvpath, 'publish_time'))
    trend_time = np.array(get_string_header(csvpath, 'trending_date'))

    time_gap = list()
    for i in range (publish_time.shape[0]):
        pt_year = int(publish_time[i][0:4])
        pt_month = int(publish_time[i][5:7])
        tt_year = int('20' + trend_time[i][0:2])
        tt_month = int(trend_time[i][6:8])
        time_gap.append((tt_year - pt_year) * 12 + (tt_month - pt_month))

