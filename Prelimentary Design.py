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

# Data preprocessing
# Training Set
train_label = kent.get_label('last_trendingdate_train.csv', view_bar, para_bar)
train_title, train_publish_time, train_category, train_tags, train_description = zlq.word_embedding('last_trendingdate_train.csv')

# Valid Set
valid_label = kent.get_label('last_trendingdate_valid.csv', view_bar, para_bar)
valid_title, valid_publish_time, valid_category, valid_tags, valid_description = zlq.word_embedding('last_trendingdate_train.csv')

#Test Set
test_label = kent.get_label('last_trendingdate_test.csv', view_bar, para_bar)
test_title, test_publish_time, test_category, test_tags, test_description = zlq.word_embedding('last_trendingdate_train.csv')

# Training - GBM


# Training - LSTM