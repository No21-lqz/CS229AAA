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


def main():
    # view = util.load_spam_dataset('spam_train.tsv', 'views', add_intercept=False)
    # like = util.load_spam_dataset('spam_train.tsv', 'likes', add_intercept=False)
    # dislike = util.load_spam_dataset('spam_train.tsv', 'dislikes', add_intercept=False)
    # comment = util.load_spam_dataset('spam_train.tsv', 'comment_count', add_intercept=False)
    # date = util.load_csv('USvideos.csv', label_col='trending_date', add_intercept=False)
    with open('last_trendingdate_train.csv', 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        train_title = [row['title'] for row in reader]
    print(train_title)

main()