import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import util
import nltk
import time

def get_para(view, like, dislike, comment):
    """
    :param view: number of view, int
    :param like: number of like, int
    :param dislike: number dislike, int
    :param comment: number of comment, int
    :return: parameter, float
    """
    percen_like = like / view
    percen_dislike = dislike / view
    percen_comment = comment / view
    parameter = (percen_like - percen_dislike) * percen_comment
    return parameter
