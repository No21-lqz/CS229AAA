import math
import util
import numpy as np
import matplotlib.pyplot as plt
import kent as kent
import LIQIAN as zlq
import lyp_preprocessing as lyp

views, likes, dislikes, comment_count = kent.load_predict_number_dataset('last_trendingdate_test.csv')
train_parameter = zlq.get_para(views, likes, dislikes, comment_count)
view_bar, para_bar = 100000, [0, 300]
label = zlq.label(views, train_parameter, view_bar, para_bar)