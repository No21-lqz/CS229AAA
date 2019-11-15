import math
import util
import numpy as np
import matplotlib.pyplot as plt
import kent as kent
import LIQIAN as zlq
import lyp_preprocessing as lyp

#Initial Boundary
view_bar, para_bar = 100000, [0, 300]
#Initial size of dictionary
size_of_dictionary = 1000
# Training Set
train_views, train_likes, train_dislikes, train_comment_count = kent.load_predict_number_dataset('last_trendingdate_train.csv')
train_parameter = zlq.get_para(train_views, train_likes, train_dislikes, train_comment_count)
train_label = zlq.label(train_views, train_parameter, view_bar, para_bar)
train_title, train_publish_time, train_category, train_tags, train_description = kent.get_feature('last_trendingdate_train.csv')
# Word embedding for training set
token_title = zlq.get_token(train_title, 'title', size_of_dictionary)
token_description = zlq.get_token(train_description, 'description', size_of_dictionary)
token_tags = zlq.get_token(train_tags, 'tags', size_of_dictionary)
one_hot_title = zlq.one_hot(train_title, size_of_dictionary)
one_hot_description = zlq.one_hot(train_description, size_of_dictionary)
one_hot_tags = zlq.one_hot(train_tags, size_of_dictionary)


# Valid Set
valid_views, valid_likes, valid_dislikes, valid_comment_count = kent.load_predict_number_dataset('last_trendingdate_valid.csv')
valid_parameter = zlq.get_para(valid_views, valid_likes, valid_dislikes, valid_comment_count)
valid_label = zlq.label(valid_views, valid_parameter, view_bar, para_bar)
valid_title, valid_publish_time, valid_category, valid_tags, valid_description = kent.get_feature('last_trendingdate_train.csv')
#Test Set
test_views, test_likes, test_dislikes, test_comment_count = kent.load_predict_number_dataset('last_trendingdate_test.csv')
test_parameter = zlq.get_para(test_views, test_likes, test_dislikes, test_comment_count)
test_label = zlq.label(test_views, test_parameter, view_bar, para_bar)
test_title, test_publish_time, test_category, test_tags, test_description = kent.get_feature('last_trendingdate_train.csv')