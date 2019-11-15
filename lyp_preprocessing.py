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

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0)
