import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from keras.preprocessing.text import Tokenizer

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


def get_token(string, header):
    """
    Word enbeding
    Function: remove the punctuation, lowercases words, and covert the words to sequences of integers
    :param string: A list of word, lenth: n
    :return: A list of integers, representing the word
    Site: https://towardsdatascience.com/recurrent-neural-networks-by-example-in-python-ffd204f99470
    """
    if header == 'tags':
        tokenizer = Tokenizer(num_words=1000,
                              filters='!@#$%^&*()_+-=\|{}[]:;">/?<,.~',
                              lower=True, split='|')
    else:
        tokenizer = Tokenizer(num_words=10000,
                              filters='!@#$%^&*()_+-=\|{}[]:;">/?<,.~',
                              lower=True)

    tokenizer.fit_on_texts(string)

    sequences = tokenizer.texts_to_sequences(string)
    print(tokenizer.index_word)
    return sequences

def one_hot(string):
    """
    One hot word embedding
    :param string: A list of strings
    :return: A matrix of integers
    """
    t = Tokenizer(num_words=1000,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True, split=' ')
    t.fit_on_texts(string)
    #print(t.index_word)
    encoded_docs = t.texts_to_matrix(string, mode='binary')
    return encoded_docs

# string = 'Jason Momoa & Lisa Bonet: Love at First Sight,The Late Late Show with James Corden'
# print(get_token(string, 'title'))