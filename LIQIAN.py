import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import util
import nltk
import time


def get_para(view, like, dislike, comment):
    """
    :param view: number of view, NumPy array shape (n_examples, 1)
    :param like: number of like, NumPy array shape (n_examples, 1)
    :param dislike: number dislike, NumPy array shape (n_examples, 1)
    :param comment: number of comment, NumPy array shape (n_examples, 1)
    :return: parameter, NumPy array shape (n_examples, 1), float
    """
    return (like - 1.5 * dislike) * comment / view

def label(view, parameter, view_bar, para_bar):
    """
        Args:
         view: number of view, NumPy array shape (n_examples, 1)
         parameter: the enmotional trend of the reflects from viewers, NumPy array shape (n_examples, 1)
         view_bar: number dislike, NumPy array shape (n_examples, 1)
         para_bar: bars of parameters, a list (2,)

    Returns:
        label, NumPy array shape (n_examples, 1), int

        """
    label = np.zeros(np.shape(view))
    n = len(view)
    [bar1, bar2] = para_bar
    for i in range(n):
        if view[i] < view_bar:
            label[i] = 0
        elif parameter[i] < bar1:
            label[i] = 1
        elif parameter[i] < bar2:
            label[i] = 2
        else:
            label[i] = 3
    return label