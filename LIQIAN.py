import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from keras.preprocessing.text import Tokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import lyp_preprocessing as lyp
import kent
import util
import collections

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


def one_hot_test(train, test, k):
    t = Tokenizer(num_words=k,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True, split=' ')
    t.fit_on_texts(train)
    encoded_docs = t.texts_to_matrix(test, mode='binary')
    return np.array(encoded_docs)


def word_embedding(csv_path, size_of_dictionary, size_of_dictionary_description):
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
    one_hot_description = util.add_intercept_fn(one_hot(description, size_of_dictionary_description))
    one_hot_tags = util.add_intercept_fn(one_hot(tags, size_of_dictionary))
    time = lyp.get_time_gap(publish_time, trending_date)
    time = util.add_intercept_fn(np.reshape(time, (len(time), 1)))
    category = util.add_intercept_fn(np.reshape(category, (len(category), 1)))
    return one_hot_title, time, category, one_hot_tags, one_hot_description


def word_embedding_test(train_path, test_path, size_of_dictionary, size_of_dictionary_description):
    train_title, train_trending_date, train_publish_time, train_category, train_tags, train_description = kent.get_feature(train_path)
    test_title, test_trending_date, test_publish_time, test_category, test_tags, test_descriotion = kent.get_feature(test_path)
    one_hot_title = util.add_intercept_fn(one_hot_test(train_title, test_title,size_of_dictionary))
    one_hot_description = util.add_intercept_fn(one_hot_test(train_description, test_descriotion, size_of_dictionary_description))
    one_hot_tags = util.add_intercept_fn(one_hot_test(train_tags, test_tags, size_of_dictionary))
    time = lyp.get_time_gap(test_publish_time, test_trending_date)
    time = util.add_intercept_fn(np.reshape(time, (len(time), 1)))
    category = util.add_intercept_fn(np.reshape(test_category, (len(test_category), 1)))
    return one_hot_title, time, category, one_hot_tags, one_hot_description


def separa_test(csv):
    """
    Seprarte the test data by publish date
    :return: three set, containing the index of the video in test set
             first set: videos trended in the train or valid set
             third set: videos published and trended in the test set
             second set: rest of the videos
    """
    new1 = []
    new3 = []
    publish_time = kent.get_time(csv)
    test_title = lyp.get_string_header(csv, 'title')
    train_title = lyp.get_string_header(csv, 'title')
    valid_title = lyp.get_string_header(csv, 'title')
    title = train_title + valid_title
    for i in range(len(publish_time)):
        pt_year = int(publish_time[i][0:4])
        pt_month = int(publish_time[i][5:7])
        pt_date = int(publish_time[i][8:10])
        if pt_year < 2018 and test_title[i] in title:
            new1 += [i]
        elif pt_year == 2018 and pt_month < 4 and test_title[i] in title:
            new1 += [i]
        elif pt_year == 2018 and pt_month == 4 and pt_date < 14 and test_title[i] in title:
            new1 += [i]
        elif pt_year == 2018 and pt_month > 4:
            new3 += [i]
        elif pt_year == 2018 and pt_month == 4 and pt_date >= 14:
            new3 += [i]
    return new1, new3


def accurancy(y_label, prediction):
    """
    Calculate the accurancy
    :param y_label: a list of true label
    :param prediction: a list of predicted label
    :return: the accurancy, float
    """
    n = len(y_label)
    result = 0
    new = np.zeros((4, ))
    for i in range(n):
        if y_label[i] == prediction[i]:
            result += 1
            t = int(y_label[i])
            new[t] += 1
    print('The accurancy count in each type', new)
    print('The count of each type:', collections.Counter(prediction))
    return result / n


def first_layer(fit_type, train_label, valid_type):
    """
    :param fit_type: Description, Title, Tags etc. a list
    :param train_label: a list of train label
    :param valid_type: a list of valid label
    :return: an array of the probability
    """
    y_train = train_label
    clf = SGDClassifier(alpha=0.2, loss="modified_huber", penalty='l2', tol=1e-6, max_iter=10000, fit_intercept=False)
    clf.fit(fit_type, y_train)
    predict = clf._predict_proba(valid_type)
    train_probability = clf._predict_proba(fit_type)
    return predict, train_probability

def GBM_model(train, test, label_train, label_test):
    """

    :param train: n x factor array, representing all factors in array
    :param test: n x factor array, representing all factors in array
    :param label_train: n x 1 array, representing the label of train
    :param label_test: n x 1 array, representing the label of test
    :return: the prediction result of GBM model
    """
    clf = GradientBoostingClassifier(max_depth = 5, tol = 0.0001)
    clf.fit(train, label_train)
    print('Finish fit')
    prediction = clf.predict(test)
    print(collections.Counter(prediction))
    print('Finish predict')
    predict_ce = clf.predict_proba(test)
    return prediction, predict_ce

def random_forest(train, y, test, label):
    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    clf.fit(train, y)
    prediction = clf.predict(test)
    acc = clf.score(test, label)
    print('accurancy:', acc)
    return prediction


