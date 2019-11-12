import numpy as np
import matplotlib.pyplot as plt
import util
# import ntlk
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


def main():
    # view = util.load_spam_dataset('spam_train.tsv', 'views', add_intercept=False)
    # like = util.load_spam_dataset('spam_train.tsv', 'likes', add_intercept=False)
    # dislike = util.load_spam_dataset('spam_train.tsv', 'dislikes', add_intercept=False)
    # comment = util.load_spam_dataset('spam_train.tsv', 'comment_count', add_intercept=False)
    date = util.load_csv('USvideos.csv', label_col='trending_date', add_intercept=False)
main()