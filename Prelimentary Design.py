import math

import sklearn

import util
import numpy as np
import matplotlib.pyplot as plt
import kent as kent
import LIQIAN as zlq
import lyp_preprocessing as lyp
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from scipy.special import softmax
import collections

#Initial Boundary
view_bar, para_bar = 100000, [0, 300]
#Initial size of dictionary
size_of_dictionary = 1000

# Data preprocessing
# Training Set
train_label = kent.get_label('last_trendingdate_train.csv', view_bar, para_bar)
train_softmax_label = kent.softmax_label('last_trendingdate_train.csv', view_bar, para_bar)
train_title, train_time, train_category, train_tags, train_description = zlq.word_embedding('last_trendingdate_train.csv',size_of_dictionary)

# Valid Set
valid_label = kent.softmax_label('last_trendingdate_valid.csv', view_bar, para_bar)
valid_title, valid_time, valid_category, valid_tags, valid_description = zlq.word_embedding('last_trendingdate_train.csv',size_of_dictionary)

#Test Set
test_label = kent.softmax_label('last_trendingdate_test.csv', view_bar, para_bar)
test_title, test_time, test_category, test_tags, test_description = zlq.word_embedding('last_trendingdate_train.csv',size_of_dictionary)

# voter
#Training - Voter
y_train = train_label
clf = SGDClassifier(alpha=0.2, loss="modified_huber", penalty="l2", max_iter=10000, fit_intercept=False)
#clf.fit(train_time, y_train)
#predict_time = clf.predict_proba(train_time)
#clf.fit(train_title, y_train)
#predict_title = clf.predict_proba(train_title)
print(train_time)
clf.fit(train_category, y_train)
predict_category = clf.predict_proba(train_category)
clf.fit(train_tags, y_train)
predict_tags = clf.predict_proba(train_tags)
clf.fit(train_description, y_train)
predict_description = clf.predict_proba(train_description)
train_p = np.array([predict_time, predict_title, predict_category, predict_tags, predict_description])
print(train_p.shape)
clf2 = SGDClassifier(alpha=0.2, loss="modified_huber", penalty="l2", max_iter=10000, fit_intercept=True)
clf2.fit(train_p, train_softmax_label)
print(clf2.coef_)
print(clf2.intercept_)



#theta_category = clf.fit(train_category, y_train)
#theta_tags = clf.fit(train_tags, y_train)
#theta_description = clf.fit(train_description, y_train)
# Training - GBM


# Training - LSTM