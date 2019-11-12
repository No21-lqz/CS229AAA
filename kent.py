import util
import os


def load_predict_number_dataset(csv_path):
    """Load dataset for view, like, dislike, comment number.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        view: NumPy array shape (n_examples, 1)
        like: NumPy array shape (n_examples, 1)
        dislike: Numpy array shape(n_examples, 1)
        comment: Numpy array shape(n_examples, 1)
    """
#util.load_dataset('USvideos', add_intercept=False)