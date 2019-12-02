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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import f1_score as f1


#from scipy.special import softmax
import collections

#Initial Boundary
view_bar, para_bar = 100000, [0, 300]
#Initial size of dictionary
size_of_dictionary = 1000

# Data preprocessing
dictionary = zlq.loadGolveModel('glove.42B.300d.txt')
# Training Set
train_label = kent.get_label('training.csv', view_bar, para_bar)
# train_softmax_label = kent.softmax_label('training.csv', view_bar, para_bar)
train_title, train_time, train_category, train_tags, train_description = zlq.word_embedding('training.csv', dictionary)


# Valid Set
valid_label = kent.get_label('valid.csv', view_bar, para_bar)
# valid_softmax_label = kent.softmax_label('last_trendingdate_valid.csv', view_bar, para_bar)
valid_title, valid_time, valid_category, valid_tags, valid_description = zlq.word_embedding('valid.csv', dictionary)

#Test Set
# test_label = kent.softmax_label('last_trendingdate_test.csv', view_bar, para_bar)
# test_title, test_time, test_category, test_tags, test_description = zlq.word_embedding('last_trendingdate_test.csv',size_of_dictionary)
# test_1, test_2, test3 = zlq.separa_test('last_trendingdate_test.csv')

train = np.hstack((train_title, train_time, train_category, train_tags, train_description))
valid = np.hstack((valid_title, valid_time, valid_category, valid_tags, valid_description))
print(np.shape(train), np.shape(valid))
# prediction, predict_ce = zlq.GBM_model(train, valid, train_label, valid_label)
# prediction = zlq.random_forest(train, train_label, valid, valid_label)
prediction = zlq.svm_prediction(train, train_label, valid)
# prediction = zlq.neuron_network(train, valid, train_label)
# fun1 = GradientBoostingClassifier(max_depth=8, tol=0.00001, random_state = 1)
# fun2 = RandomForestClassifier(n_estimators=150)
# fun3 = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-5, hidden_layer_sizes=(20, 5))
# prediction = zlq.vote(fun3, fun2, fun1, train, valid, train_label)

np.savetxt('Combination_1.txt', prediction)
baseline = 2 * np.ones((5743, ))
f1_score = f1(valid_label, prediction, average='weighted')
base_f1 = f1(valid_label, baseline, average='weighted')
acc = zlq.accurancy(valid_label, prediction)
print('f1_score:', f1_score)
print('f1_base:', base_f1)
print(acc)


























# voter
#Training - Voter
#
# n = train_title.shape[0]
# y_train = train_label
# normalize_time = lyp.normalization(train_time)
#
# clf = SGDClassifier(alpha=0.2, loss="modified_huber", penalty="l2", max_iter=10000, tol=1e-6, fit_intercept=False)
# clf.fit(train_time, y_train)
# pro_time = clf.predict_proba(train_time)
# predict_time = clf.predict_proba(valid_time)
#
#
# clf.fit(train_title, y_train)
# pro_title = clf.predict_proba(train_title)
# predict_title = clf.predict_proba(valid_title)
#

# clf.fit(train_category, y_train)
# pro_category = clf.predict_proba(train_category)
# predict_category = clf.predict_proba(valid_category)
#
#
# clf.fit(train_tags, y_train)
# pro_tags = clf.predict_proba(train_tags)
# predict_tags = clf.predict_proba(valid_tags)
#
#
# clf.fit(train_description, y_train)
# pro_description = clf.predict_proba(train_description)
# predict_description = clf.predict_proba(valid_description)
#
# train_p = lyp.create_3d(pro_time, pro_title, pro_category, pro_tags, pro_description)
#
# theta = lyp.train(train_p, train_softmax_label)
#
# predict_p = lyp.create_3d(predict_time, predict_title, predict_category, predict_tags, predict_description)
# predict = lyp.predict_pro(predict_p, theta)
#predict_label = lyp.predict(predict)
# acc = lyp.accuracy(predict, valid_label)

#
# print(theta)
# print(acc)
#
# print(collections.Counter(predict.ravel()))

#theta_category = clf.fit(train_category, y_train)
#theta_tags = clf.fit(train_tags, y_train)
#theta_description = clf.fit(train_description, y_train)
# Training - GBM


# Training - LSTM
