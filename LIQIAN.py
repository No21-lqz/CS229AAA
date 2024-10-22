import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score as f1
from keras.preprocessing.text import Tokenizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import svm
import warnings
from sklearn.model_selection import GridSearchCV
import lyp_preprocessing as lyp
import kent
import util
from sklearn.tree import DecisionTreeClassifier
import collections
from gensim.models import KeyedVectors
#from xgboost import XGBClassifier
import mord
import re

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


def loadGolveModel(glove_file):
    f = open(glove_file, 'r', encoding='UTF-8')
    model = {}
    for line in f:
        splitline = line.split()
        word = splitline[0].replace("'", "")
        embedding = np.array([float(val) for val in splitline[1: ]])
        model[word] = embedding
    print("Done.", len(model), "words loaded!")
    return model


def load_index_dic(glove_file):
    f = open(glove_file, 'r', encoding='UTF-8')
    dic = []
    for line in f:
        splitline = line.split()
        dic.append(splitline[0])
    f.close()
    return dic


def glove_embedding_one_string(string, dictionary):
    words = string.lower().split()
    new_words = [re.sub('[{}!#?,.:";@$%^&*()_+-=|[]:;">/?<,.~]', '', word) for word in words]
    temp = [dictionary[i] for i in new_words if i in dictionary.keys()]
    temp = np.array(temp)
    return np.sum(temp, axis=0)


def glove_embedding(list, dictionary):
    n, t = len(list), 0
    l = dictionary['a'].shape[0]
    temp = np.zeros((n, l))
    for i in list:
        temp[t] = glove_embedding_one_string(i, dictionary)
        t += 1
    return np.array(temp)


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


def word_embedding(csv_path, dictionary):
    """
    Get the structured input data
    :param csv_path: The trina,valid, and test test path, .csv file name
    :param size_of_dictionary:  a int
    :return: structured title, tag, description, list type, each with a lenth of dictionary,
             category as integer, publish_time as time
             Type: np.array
    """
    title, trending_date, publish_time, category, tags, description, duration = kent.get_feature(csv_path)
    glove_title = glove_embedding(title, dictionary)
    glove_description = glove_embedding(description, dictionary)
    glove_tags = glove_embedding(tags, dictionary)
    time = lyp.get_time_gap(publish_time, trending_date)
    category = util.add_intercept_fn(np.reshape(category, (len(category), 1)))
    time = time.reshape((len(time), 1))
    duration = duration.reshape((len(duration), 1))
    return glove_title, time, category, glove_tags, glove_description, duration


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


def GBM_model(train, train_label, test, test_label):
    """

    :param train: n x factor array, representing all factors in array
    :param test: n x factor array, representing all factors in array
    :param label_train: n x 1 array, representing the label of train
    :param label_test: n x 1 array, representing the label of test
    :return: the prediction result of GBM model
    """
    model = GradientBoostingClassifier(max_depth=5, tol=0.0001, n_estimators=100)
    eval_set = [(train, train_label), (test, test_label)]
    model.fit(train, train_label, eval_metric=["merror", "mlogloss"], eval_set=eval_set, verbose=True)
    print('Finish GBM fit')
    prediction = model.predict(test)
    print('Finish GBM prediction')
    return prediction


def GBM_multi_model(train, train_label, test):
    """

    :param train: n x factor array, representing all factors in array
    :param test: n x factor array, representing all factors in array
    :param label_train: n x 1 array, representing the label of train
    :param label_test: n x 1 array, representing the label of test
    :return: the prediction result of GBM model
    """
    # w_array = np.array([0.7] * train_label.shape[0])
    # w_array[train_label == 0] = 0.9
    # w_array[train_label == 1] = 8
    # w_array[train_label == 3] = 1.7
    model = GradientBoostingClassifier(max_depth=8, tol=0.0001, n_estimators=100)
    model.fit(train, train_label)
    print('Finish GBM fit')
    prediction = model.predict(test)
    print('Finish GBM prediction')
    return prediction

def random_forest(train, train_label, test):
    clf = RandomForestClassifier(random_state=27 ,max_features=None, n_estimators=300,
                                 class_weight={0:2.92, 1:65, 2:1, 3:7.4})
    clf.fit(train, train_label)
    prediction = clf.predict(test)
    return prediction

def random_forest_multi(train, train_label, test):
    clf = RandomForestClassifier(random_state=27 ,max_features=None, n_estimators=300)
    clf.fit(train, train_label)
    prediction = clf.predict(test)
    return prediction


def neuron_network(train, label_train, test):
    clf = MLPClassifier(solver='adam', activation='logistic', alpha=0.4, tol=1e-5,
                        hidden_layer_sizes=(100, 20), max_iter=500)
    clf.fit(train, label_train)
    prediction = clf.predict(test)
    return prediction


def vote(fun1, fun2, fun3, train, train_label, valid):
    clf = VotingClassifier(estimators=[('fun1', fun1), ('fun2', fun2), ('fun3', fun3)], voting='hard')
    clf.fit(train, train_label)
    prediction = clf.predict(valid)
    return prediction


def svm_prediction(train, train_label, test):
    clf = svm.SVC(C=1.0, cache_size=200, coef0=1.0,
        decision_function_shape='ovo', degree=5, gamma='scale', kernel='poly',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=True)
    clf.fit(train, train_label)
    prediction = clf.predict(test)
    return prediction

#
# def mord_predict(train, train_label, test):
#     clf = mord.MulticlassLogistic()
#     clf.fit(train, train_label)
#     prediction = clf.predict(test)
#     return prediction
#
# def xgb_prediction(train, train_label, test):
#     clf = XGBClassifier(booster = "gbtree")  #objective = reg:squaredlogerror
#     clf.fit(train, train_label)
#     return clf.predict(test)

def tree(train, train_label, test, i):
    clf = DecisionTreeClassifier(random_state=i, class_weight={0:5, 1:5, 2:0.05, 3:1})  #, class_weight={0:1, 1:1, 2:1, 3:1}
    clf.fit(train, train_label)
    prediction = clf.predict(test)
    return prediction

def tree_multi(train, train_label, test):
    clf = DecisionTreeClassifier()  #, class_weight={0:1, 1:1, 2:1, 3:1}
    clf.fit(train, train_label)
    prediction = clf.predict(test)
    return prediction


def relable(label, target_label):
    """
    change the multiple class into binary class
    :param label: the array of the original label
    :param target_label:
    :return: an array of the label, 1 means label is the targeted one and 0 is other labels
    """
    return np.array([int(i == target_label) for i in label])


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy


def sgdc(train, train_label, test, random):
    clf = SGDClassifier(random_state=random, alpha=0.2, loss="modified_huber", penalty='l2', tol=1e-6, max_iter=10000, fit_intercept=False)
    clf.fit(train, train_label)
    predict = clf.predict(test)
    return predict

def sgdc_multi(train, train_label, test):
    clf = SGDClassifier(alpha=7.5, loss="modified_huber", penalty='l2', tol=1e-6, fit_intercept=False)
    clf.fit(train, train_label)
    predict = clf.predict(test)
    return predict


def delete_feature(train, function, train_label, test, test_label, name, random):
    """
    :param list: list of separate feature
    :param function: the training model
    :return:
    """
    def g(train, test, name):
        # Get the f1 score
        n = len(train)
        f1_score = np.zeros((n,))
        temp_name = name
        c = []
        if n == 1:
            # print('The last class:', name[0])
            return None
        for i in range(n):
            temp_train, temp_test = train.copy(), test.copy()
            temp_train.pop(i)
            temp_test.pop(i)
            new_train = temp_train[0]
            new_test = temp_test[0]
            if n - 2 > 0:
                for j in range(n - 2):
                    new_train = np.hstack((new_train, temp_train[j + 1]))
                    new_test = np.hstack((new_test, temp_test[j + 1]))
            prediction = function(new_train, train_label, new_test, random)
            c += [collections.Counter(prediction)]
            warnings.filterwarnings('ignore')
            f1_score[i] = f1(test_label, prediction, average='weighted')
            # print("the f1 score with class", name[i], "excluded:", f1_score[i])
        remain_class = np.argmax(f1_score)
        del name[remain_class]
        train.pop(remain_class)
        test.pop(remain_class)
        print('The remaining class is:', temp_name)
        print('the class predicted is:', c[remain_class])
        return delete_feature(train, function, train_label, test, test_label, name, random)

    return g(train, test, name)
