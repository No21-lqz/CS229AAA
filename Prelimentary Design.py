import math
import util
import numpy as np
import matplotlib.pyplot as plt
import kent as kent
import LIQIAN as zlq
import lyp_preprocessing as lyp
from sklearn import ensemble
from sklearn import linear_model

#Initial Boundary
view_bar, para_bar = 100000, [0, 300]

# Training Set
train_views, train_likes, train_dislikes, train_comment_count = kent.load_predict_number_dataset('last_trendingdate_train.csv')
train_parameter = zlq.get_para(train_views, train_likes, train_dislikes, train_comment_count)
train_label = zlq.label(train_views, train_parameter, view_bar, para_bar)
train_title = lyp.get_string_header('last_trendingdate_train.csv', 'title')
train_publish_time = lyp.get_string_header('last_trendingdate_train.csv', 'publish_time')
train_category = kent.load_number_dataset('last_trendingdate_train.csv', 'category_id')
train_tags = lyp.get_string_header('last_trendingdate_train.csv', 'tags')
train_description = lyp.get_string_header('last_trendingdate_train.csv', 'description')

# Valid Set
valid_views, valid_likes, valid_dislikes, valid_comment_count = kent.load_predict_number_dataset('last_trendingdate_valid.csv')
valid_parameter = zlq.get_para(valid_views, valid_likes, valid_dislikes, valid_comment_count)
valid_label = zlq.label(valid_views, valid_parameter, view_bar, para_bar)
valid_title = lyp.get_string_header('last_trendingdate_valid.csv', 'title')
valid_publish_time = lyp.get_string_header('last_trendingdate_valid.csv', 'publish_time')
valid_category = kent.load_number_dataset('last_trendingdate_valid.csv', 'category_id')
valid_tags = lyp.get_string_header('last_trendingdate_valid.csv', 'tags')
valid_description = lyp.get_string_header('last_trendingdate_valid.csv', 'description')

#Test Set
test_views, test_likes, test_dislikes, test_comment_count = kent.load_predict_number_dataset('last_trendingdate_test.csv')
test_parameter = zlq.get_para(test_views, test_likes, test_dislikes, test_comment_count)
test_label = zlq.label(test_views, test_parameter, view_bar, para_bar)
test_title = lyp.get_string_header('last_trendingdate_test.csv', 'title')
test_publish_time = lyp.get_string_header('last_trendingdate_test.csv', 'publish_time')
test_category = kent.load_number_dataset('last_trendingdate_test.csv', 'category_id')
test_tags = lyp.get_string_header('last_trendingdate_test.csv', 'tags')
test_description = lyp.get_string_header('last_trendingdate_test.csv', 'description')