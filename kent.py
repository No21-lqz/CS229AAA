import util
import os
import numpy as np
import pandas as pd
import lyp_preprocessing as lyp
import LIQIAN as zlq


def load_predict_number_dataset(csv_path):
    """Load dataset for view, like, dislike, comment number.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        view: NumPy array shape (n_examples, 1)
        like: NumPy array shape (n_examples, 1)
        dislike: Numpy array shape(n_examples, 1)
        comment: Numpy array shape(n_examples, 1)
    """
    # Load headers
    with open(csv_path, encoding='gb18030', errors='ignore', newline='') as csv_fh:
        headers = csv_fh    .readline().strip().split(',')
    # Load features and labels
    view_cols = [i for i in range(len(headers)) if headers[i] == 'views']
    likes_cols = [i for i in range(len(headers)) if headers[i] == 'likes']
    dislikes_cols = [i for i in range(len(headers)) if headers[i] == 'dislikes']
    comment_cols = [i for i in range(len(headers)) if headers[i] == 'comment_count']
    views = np.array(pd.read_csv(csv_path, usecols = view_cols)).ravel()
    likes = np.array(pd.read_csv(csv_path, usecols = likes_cols)).ravel()
    dislikes = np.array(pd.read_csv(csv_path, usecols = dislikes_cols)).ravel()
    comment_count = np.array(pd.read_csv(csv_path, usecols = comment_cols)).ravel()

    return views, likes, dislikes, comment_count

def load_number_dataset(csv_path, header):
    """Load dataset for view, like, dislike, comment number.

    Args:
         csv_path: Path to CSV file containing dataset.
         header: input the header of column (type: string)

    Returns:
        view: NumPy array shape (n_examples, 1)
        like: NumPy array shape (n_examples, 1)
        dislike: Numpy array shape(n_examples, 1)
        comment: Numpy array shape(n_examples, 1)
    """
    # Load headers
    with open(csv_path, encoding='gb18030', errors='ignore', newline='') as csv_fh:
        headers = csv_fh    .readline().strip().split(',')
    # Load features and labels
    cols = [i for i in range(len(headers)) if headers[i] == header]
    data = np.array(pd.read_csv(csv_path, usecols = cols)).ravel()


    return data

def get_feature(cvs_path):
    """

    :param: cvs_path: Path to CSV file containing dataset
    :return:
        title: a list of string data you want (dim: n)
        trending_date: a list of string data you want (dim: n)
        publish_time: a list of string data you want (dim: n)
        category:a list of string data you want (dim: n)
        tags:a list of string data you want (dim: n)
        description:a list of string data you want (dim: n)
    """
    title = lyp.get_string_header('last_trendingdate_train.csv', 'title')
    trending_date = lyp.get_string_header('last_trendingdate_train.csv', 'trending_date')
    publish_time = lyp.get_string_header('last_trendingdate_train.csv', 'publish_time')
    category = load_number_dataset('last_trendingdate_train.csv', 'category_id')
    tags = lyp.get_string_header('last_trendingdate_train.csv', 'tags')
    description = lyp.get_string_header('last_trendingdate_train.csv', 'description')
    return title,trending_date, publish_time, category, tags, description

def get_label(csvpath, view_bar, para_bar):
    """
    :param csvpath: name of csv flie (type: string, like '***.csv')
    :param view_bar: number dislike, NumPy array shape (n_examples, 1)
    :param para_bar: bars of parameters, a list (2,)
    :return:
        Label, NumPy array shape (n_examples, 1), int
        0: Not hot
        1: Negative, dislike >> like
        2: Controdictory, dislike ~= like
        3: Positive, like >> dislike
    """
    views, likes, dislikes, comment_count = load_predict_number_dataset(csvpath)
    parameter = zlq.get_para(views, likes, dislikes, comment_count)
    label = zlq.label(views, parameter, view_bar, para_bar)
    return label


def softmax_label(csvpath, view_bar, para_bar):
    label = get_label(csvpath, view_bar, para_bar)
    new = np.zeros((len(label), 4))
    for i in range(len(label)):
        index = int(label[i])
        new[i][index] = 1
    return new
