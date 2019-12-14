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
train_label = kent.get_label('training_valid_set_with_time.csv', view_bar, para_bar)
train_title, train_time, train_category, train_tags, train_description, train_duration = zlq.word_embedding('training_valid_set_with_time.csv', dictionary)


# Test
test_label = kent.get_label('test_set_with_time.csv', view_bar, para_bar)
test_title, test_time, test_category, test_tags, test_description, test_duration = zlq.word_embedding('test_set_with_time.csv', dictionary)


train = np.hstack((train_time, train_category, train_description))
test = np.hstack((test_time, test_category, test_description))


# PCA Analysis
pca = PCA(n_components=20)
pca.fit(train)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)

transformer = KernelPCA(n_components=2, kernel='rbf')
train_transformed = transformer.fit_transform(train)
test_transformed = transformer.fit_transform(test)
ss = StandardScaler()
std_train = ss.fit_transform(train_transformed)
std_valid = ss.fit_transform(test_transformed)
util.plot(train_transformed, train_label, 'train_std.png')



order = np.array([2, 0, 3, 1])

#Model
uniform_base = np.random.choice(4, test_label.shape)
base_f1 = f1(test_label, uniform_base, average='weighted')
print('f1_score of base line:', base_f1)

prediction_sgdc = zlq.sgdc(train, train_label, test, 0)
np.savetxt('SGDC prediction.txt', prediction_sgdc)
zlq.accurancy(test_label, prediction_sgdc)
f1_score = f1(test_label, prediction_sgdc, average='weighted')
print('The f1 score of SGDC model is', f1_score)

prediction_xgb = lyp.xgb_prediction(train, train_label, test, test_label)
np.savetxt('xgb_prediction_with_time.txt', prediction_xgb)
f1_score = f1(test_label, prediction_xgb, average='weighted')
zlq.accurancy(test_label, prediction_xgb)
print('f1_score_xgb:', f1_score)

prediction_gbm = zlq.GBM_model(train, train_label, test)
np.savetxt('gbm_prediction_with_time.txt', prediction_gbm)
f1_score = f1(test_label, prediction_gbm, average='weighted')
zlq.accurancy(test_label, prediction_gbm)
print('f1_score_gbm:', f1_score)

prediction_randomforest = zlq.random_forest(train, train_label, test)
np.savetxt('randomforest_prediction_with_time.txt', prediction_randomforest)
f1_score = f1(test_label, prediction_randomforest, average='weighted')
zlq.accurancy(test_label, prediction_randomforest)
print('f1_score_randomforest:', f1_score)


#Multi-layer Binary Classifier
prediction_random = kent.multibinary(train, train_label, test, zlq.random_forest_multi, order)
np.savetxt('Combination_random_with_time.txt', prediction_random)
print(collections.Counter(prediction_random))
f1_score = f1(test_label, prediction_random, average='weighted')
zlq.accurancy(test_label, prediction_random)
print('f1_score_random:', f1_score)

prediction_GBM_model = kent.multibinary(train, train_label, test, zlq.GBM_multi_model, order)
np.savetxt('Combination_GBM_model_with_time.txt', prediction_GBM_model)
print(collections.Counter(prediction_GBM_model))
f1_score = f1(test_label, prediction_GBM_model, average='weighted')
zlq.accurancy(test_label, prediction_GBM_model)
print('f1_score_GBM:', f1_score)

prediction_neuron_network = kent.multibinary(train, train_label, test, zlq.neuron_network, order)
np.savetxt('Combination_neuron_network_with_time.txt', prediction_neuron_network)
print(collections.Counter(prediction_neuron_network))
f1_score = f1(test_label, prediction_neuron_network, average='weighted')
zlq.accurancy(test_label, prediction_neuron_network)
print('f1_score_neuron:', f1_score)

prediction_tree = kent.multibinary(train, train_label, test, zlq.tree_multi, order)
np.savetxt('Combination_tree_with_time.txt', prediction_tree)
print(collections.Counter(prediction_tree))
f1_score = f1(test_label, prediction_tree, average='weighted')
zlq.accurancy(test_label, prediction_tree)
print('f1_score_tree:', f1_score)

prediction_sgdc = kent.multibinary(train, train_label, test, zlq.sgdc_multi, order)
np.savetxt('Combination_sgdc_with_time.txt', prediction_sgdc)
print(collections.Counter(prediction_sgdc))
f1_score = f1(test_label, prediction_sgdc, average='weighted')
zlq.accurancy(test_label, prediction_sgdc)
print('f1_score_sgdc:', f1_score)

prediction_xgb = kent.multibinary(train, train_label,test,lyp.xgb_prediction_mutli, order)
np.savetxt('Combination_xgb_with_time.txt', prediction_xgb)
print(collections.Counter(prediction_xgb))
f1_score = f1(test_label, prediction_xgb, average='weighted')
zlq.accurancy(test_label, prediction_xgb)
print('f1_score_xgb:', f1_score)


# Backward search
def multi_round(fun):
    for i in range(5, 15):
        print('Random seed: ', i)
        name = ['title', 'time gap', 'category', 'tags', 'description', 'duration']
        train = [train_title, train_time, train_category, train_tags, train_description, train_duration]
        valid = [valid_title, valid_time, valid_category, valid_tags, valid_description, valid_duration]
        zlq.delete_feature(train, fun, train_label, valid, valid_label, name, i)
    return None


print('SDGC Model')
count = np.zeros((6,))
multi_round(zlq.sgdc)
print(count)

print('NN')
count = np.zeros((6,))
multi_round(zlq.neuron_network)
print(count)

print('Decision Tree')
count = np.zeros((6,))
multi_round(zlq.tree)
print(count)

print('Random Forest')
count = np.zeros((6,))
multi_round(zlq.random_forest)
print(count)

print('XGB Model')
count = np.zeros((6,))
multi_round(lyp.xgb_test)
print(count)

print('GBM')
count = np.zeros((6,))
multi_round(zlq.GBM_model)
print(count)

print('Random Forest')
count = np.zeros((6,))
multi_round(zlq.random_forest)
print(count)
