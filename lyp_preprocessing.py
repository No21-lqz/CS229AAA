import numpy as np
import matplotlib.pyplot as plt
import util
#import ntlk
import time
import csv

def get_string_header(csvpath, header):
    """
    :param csvpath: name of csv flie (type: string, like '***.csv')
    :param header: input the header of column (type: string)
    :return: a list of string data you want (dim: n)
    """
    with open(csvpath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        output = [row[header] for row in reader]
    return output