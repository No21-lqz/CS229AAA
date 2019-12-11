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
from imblearn.over_sampling import RandomOverSampler


#from scipy.special import softmax
import collections

#Initial Boundary
view_bar, para_bar = 100000, [0, 300]
#Initial size of dictionary
size_of_dictionary = 1000

# Data preprocessing
dictionary = zlq.loadGolveModel('glove.twitter.27B.25d.txt')
# Training Set
train_label = kent.get_label('training_set_with_time.csv', view_bar, para_bar)
train_title, train_time, train_category, train_tags, train_description, train_duration = zlq.word_embedding('training_set_with_time.csv', dictionary)


# Valid Set
valid_label = kent.get_label('valid_set_with_time.csv', view_bar, para_bar)
valid_title, valid_time, valid_category, valid_tags, valid_description, valid_duration = zlq.word_embedding('valid_set_with_time.csv', dictionary)

#Test Set
#test_label = kent.get_label('test.csv', view_bar, para_bar)
#test_title, test_time, test_category, test_tags, test_description = zlq.word_embedding('test.csv', dictionary)



train = np.hstack((train_title, train_time, train_category, train_tags, train_description, train_duration))

valid = np.hstack((valid_title, valid_time, valid_category, valid_tags, valid_description, valid_duration))

# order = np.array([2, 0, 3, 1])

order = np.array([0,2,1,3])

prediction_random = kent.multibinary(train, train_label, valid,zlq.random_forest_multi, order)
# np.savetxt('Combination_random_with_time.txt', prediction_random)
print(collections.Counter(prediction_random))
f1_score = f1(valid_label, prediction_random, average='weighted')
acc = zlq.accurancy(valid_label, prediction_random)
print('f1_score_random:', f1_score)
print(acc)


prediction_GBM_model = kent.multibinary(train, train_label, valid,zlq.GBM_multi_model, order)
# np.savetxt('Combination_GBM_model_with_time.txt', prediction_GBM_model)
print(collections.Counter(prediction_GBM_model))
f1_score = f1(valid_label, prediction_GBM_model, average='weighted')
acc = zlq.accurancy(valid_label, prediction_GBM_model)
print('f1_score_GBM:', f1_score)
print(acc)
#
prediction_neuron_network = kent.multibinary(train, train_label, valid,zlq.neuron_network, order)
# np.savetxt('Combination_neuron_network_with_time.txt', prediction_neuron_network)
print(collections.Counter(prediction_neuron_network))
f1_score = f1(valid_label, prediction_neuron_network, average='weighted')
acc = zlq.accurancy(valid_label, prediction_neuron_network)
print('f1_score_neuron:', f1_score)
print(acc)

# prediction_svm_prediction = kent.multibinary(train, train_label, valid,zlq.svm_prediction, order)
# # np.savetxt('Combination_svm_prediction_with_time.txt', prediction_svm_prediction)
# print(collections.Counter(prediction_svm_prediction))
# f1_score = f1(valid_label, prediction_svm_prediction, average='weighted')
# acc = zlq.accurancy(valid_label, prediction_svm_prediction)
# print('f1_score_svm:', f1_score)
# print(acc)
# #
prediction_tree = kent.multibinary(train, train_label, valid,zlq.tree_multi, order)
# np.savetxt('Combination_tree_with_time.txt', prediction_tree)
print(collections.Counter(prediction_tree))
f1_score = f1(valid_label, prediction_tree, average='weighted')
acc = zlq.accurancy(valid_label, prediction_tree)
print('f1_score_tree:', f1_score)
print(acc)
#

prediction_sgdc = kent.multibinary(train, train_label, valid,zlq.sgdc_multi, order)
# np.savetxt('Combination_sgdc_with_time.txt', prediction_sgdc)
print(collections.Counter(prediction_sgdc))
f1_score = f1(valid_label, prediction_sgdc, average='weighted')
acc = zlq.accurancy(valid_label, prediction_sgdc)
print('f1_score_sgdc:', f1_score)
print(acc)



prediction_xgb = kent.multibinary(train, train_label,valid,lyp.xgb_prediction_mutli, order)
# np.savetxt('Combination_xgb_with_time.txt', prediction_xgb)
print(collections.Counter(prediction_xgb))
f1_score = f1(valid_label, prediction_xgb, average='weighted')
acc = zlq.accurancy(valid_label, prediction_xgb)
print('f1_score_xgb:', f1_score)
print(acc)




#prediction = zlq.mord_predict(train, train_label, valid)
#prediction = zlq.svm_prediction(train_rps_x, train_rps_y, valid)
# prediction = zlq.neuron_network(train, valid, train_label)
#fun1 = GradientBoostingClassifier(max_depth=8, tol=0.00001, random_state = 1)
#fun2 = RandomForestClassifier(n_estimators=150)
#fun3 = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-5, hidden_layer_sizes=(20, 5))
#prediction = zlq.vote(fun3, fun2, fun1, train, valid, train_label)

#base_f1 = f1(valid_label, baseline, average='weighted')


'''prediction_gbm = zlq.GBM_model(train, train_label, valid)
np.savetxt('gbm_prediction_with_time.txt', prediction_gbm)
f1_score = f1(valid_label, prediction_gbm, average='weighted')
print('f1_score_gbm:', f1_score)

prediction_randomforest = zlq.random_forest(train, train_label, valid)
np.savetxt('randomforest_prediction_with_time.txt', prediction_randomforest)
f1_score = f1(valid_label, prediction_randomforest, average='weighted')
print('f1_score_randomforest:', f1_score)'''

























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
