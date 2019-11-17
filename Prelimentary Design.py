import math

import sklearn

import util
import numpy as np
import matplotlib.pyplot as plt
import kent as kent
import LIQIAN as zlq
import lyp_preprocessing as lyp
from sklearn.linear_model import SGDClassifier
import collections
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
valid_label = kent.get_label('last_trendingdate_valid.csv', view_bar, para_bar)
valid_softmax_label = kent.softmax_label('last_trendingdate_valid.csv', view_bar, para_bar)
valid_title, valid_time, valid_category, valid_tags, valid_description = zlq.word_embedding('last_trendingdate_valid.csv',size_of_dictionary)

#Test Set
test_label = kent.softmax_label('last_trendingdate_test.csv', view_bar, para_bar)
test_title, test_time, test_category, test_tags, test_description = zlq.word_embedding('last_trendingdate_test.csv',size_of_dictionary)
test_1, test_2, test3 = zlq.separa_test('last_trendingdate_test.csv')


# voter
#Training - Voter

n = train_title.shape[0]
y_train = train_label
normalize_time = lyp.normalization(train_time)

clf = SGDClassifier(alpha=0.2, loss="modified_huber", penalty="l2", max_iter=10000, tol=1e-6, fit_intercept=False)
clf.fit(train_time, y_train)
pro_time = clf.predict_proba(train_time)
predict_time = clf.predict_proba(valid_time)


clf.fit(train_title, y_train)
pro_title = clf.predict_proba(train_title)
predict_title = clf.predict_proba(valid_title)


clf.fit(train_category, y_train)
pro_category = clf.predict_proba(train_category)
predict_category = clf.predict_proba(valid_category)


clf.fit(train_tags, y_train)
pro_tags = clf.predict_proba(train_tags)
predict_tags = clf.predict_proba(valid_tags)


clf.fit(train_description, y_train)
pro_description = clf.predict_proba(train_description)
predict_description = clf.predict_proba(valid_description)

train_p = lyp.create_3d(pro_time, pro_title, pro_category, pro_tags, pro_description)

theta = lyp.train(train_p, train_softmax_label)

predict_p = lyp.create_3d(predict_time, predict_title, predict_category, predict_tags, predict_description)
predict = lyp.predict_pro(predict_p, theta)
#predict_label = lyp.predict(predict)
acc = lyp.accuracy(predict, valid_label)


print(theta)
print(acc)

print(collections.Counter(predict.ravel()))

#theta_category = clf.fit(train_category, y_train)
#theta_tags = clf.fit(train_tags, y_train)
#theta_description = clf.fit(train_description, y_train)
# Training - GBM


# Training - LSTM