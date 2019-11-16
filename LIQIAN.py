import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from keras.preprocessing.text import Tokenizer
import lyp_preprocessing as lyp
import kent
import util

def get_para(view, like, dislike, comment):
    """
    :param view: number of view, NumPy array shape (n_examples, 1)
    :param like: number of like, NumPy array shape (n_examples, 1)
    :param dislike: number dislike, NumPy array shape (n_examples, 1)
    :param comment: number of comment, NumPy array shape (n_examples, 1)
    :return: parameter, NumPy array shape (n_examples, 1), float
    """
    return (like - 1.5 * dislike) * comment / view

def label(view, parameter, view_bar, para_bar):
    """
        Args:
         view: number of view, NumPy array shape (n_examples, 1)
         parameter: the enmotional trend of the reflects from viewers, NumPy array shape (n_examples, 1)
         view_bar: number dislike, NumPy array shape (n_examples, 1)
         para_bar: bars of parameters, a list (2,)

    Returns:
        label, NumPy array shape (n_examples, 1), int
        0: Not hot
        1: Negative, dislike >> like
        2: Controdictory, dislike ~= like
        3: Positive, like >> dislike
        """
    label = np.zeros(np.shape(view))
    n = len(view)
    [bar1, bar2] = para_bar
    for i in range(n):
        if view[i] < view_bar:
            label[i] = 0
        elif parameter[i] < bar1:
            label[i] = 1
        elif parameter[i] < bar2:
            label[i] = 2
        else:
            label[i] = 3
    return label


def get_token(string, header, k):
    """
    Word embedding for token
    Function: remove the punctuation, lowercases words, and covert the words to sequences of integers
    :param string: A list of word, lenth: n
           header: type of string
           k: size of dictionary
    :return: A list of integers, representing the word
    Site: https://towardsdatascience.com/recurrent-neural-networks-by-example-in-python-ffd204f99470
    """
    if header == 'tags':
        tokenizer = Tokenizer(num_words=k,    # Word with top k frequency
                              filters='!@#$%^&*()_+-=\|{}[]:;">/?<,.~',
                              lower=True, split='|')
    else:
        tokenizer = Tokenizer(num_words=k,
                              filters='!@#$%^&*()_+-=\|{}[]:;">/?<,.~',
                              lower=True)

    tokenizer.fit_on_texts(string)
    sequences = tokenizer.texts_to_sequences(string)
    #print(tokenizer.index_word)       # print dictionary create
    return sequences

def one_hot(string, k):
    """
    One hot word embedding
    :param string: A list of strings
           k: size of dictionary
    :return: A matrix of integers reflecting the string
             dim: n-examples x m-size of dictionary
             Type: np.array
    """
    t = Tokenizer(num_words=k,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True, split=' ')
    t.fit_on_texts(string)
    encoded_docs = t.texts_to_matrix(string, mode='binary')
    return np.array(encoded_docs)


def word_embedding(csv_path, size_of_dictionary):
    """
    Get the structured input data
    :param csv_path: The trina,valid, and test test path, .csv file name
    :param size_of_dictionary:  a int
    :return: structured title, tag, description, list type, each with a lenth of dictionary,
             category as integer, publish_time as time
             Type: np.array
    """
    title,trending_date, publish_time, category, tags, description = kent.get_feature(csv_path)
    one_hot_title = util.add_intercept_fn(one_hot(title, size_of_dictionary))
    one_hot_description = util.add_intercept_fn(one_hot(description, size_of_dictionary))
    one_hot_tags = util.add_intercept_fn(one_hot(tags, size_of_dictionary))
    time = lyp.get_time_gap(publish_time, trending_date)
    time = util.add_intercept_fn(np.reshape(time, (len(time), 1)))
    category = util.add_intercept_fn(category)
    return one_hot_title, time, category, one_hot_tags, one_hot_description


def separa_test(publish_time):
    """
    Seprarte the test data by publish date
    :return: three set, containing the index of the video in test set
             first set: videos trended in the train or valid set
             third set: videos published and trended in the test set
             second set: rest of the videos
    """
    new1 = []
    new2 = []
    new3 = []
    test_title = lyp.get_string_header('last_trendingdate_test.csv', 'title')
    train_title = lyp.get_string_header('last_trendingdate_train.csv', 'title')
    valid_title = lyp.get_string_header('last_trendingdate_valid.csv', 'title')
    title = train_title.append(valid_title)
    for i in range(len(publish_time)):
        pt_year = int(publish_time[i][0:4])
        pt_month = int(publish_time[i][5:7])
        if pt_year <= 2017 and pt_month < 11 and test_title in title:
            new1 = new1.append([i])
    return




# train_tags = lyp.get_string_header('last_trendingdate_train.csv', 'tags')
# token_tags = one_hot(train_tags, 100)
# print(np.shape(token_tags))


