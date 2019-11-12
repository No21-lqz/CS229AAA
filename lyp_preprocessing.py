import numpy as np
import matplotlib.pyplot as plt
import util
#import ntlk
import time
import csv

def get_para(view, like, dislike, comment):
    percen_like = like / view
    percen_dislike = dislike / view
    percen_comment = comment / view
    parameter = (percen_like - percen_dislike) * percen_comment
    return parameter

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

def main():
    # view = util.load_spam_dataset('spam_train.tsv', 'views', add_intercept=False)
    # like = util.load_spam_dataset('spam_train.tsv', 'likes', add_intercept=False)
    # dislike = util.load_spam_dataset('spam_train.tsv', 'dislikes', add_intercept=False)
    # comment = util.load_spam_dataset('spam_train.tsv', 'comment_count', add_intercept=False)
    # date = util.load_csv('USvideos.csv', label_col='trending_date', add_intercept=False)

    print(train_title)
    print(np.size(train_title))

main()