# This is greatly inspired from the implementation of Mathieu Dumoulin
# https://github.com/dumoulma/fic-prototype
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.sparse as sp
import numpy as np


class EBNSTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, ppf_limit=(0.0005, 1 - 0.0005)):
        self.ppf_limit = ppf_limit
        self.ebns_scores = []
        self.scoring_function = lambda x: np.max(x, axis=0)

    def fit(self, X, y):
        y = np.array(y)
        if hasattr(X, 'dtype') and np.float64 == np.dtype(X).type:
            X = sp.csr_matrix(X, copy=True)
        else:
            X = sp.csr_matrix(X, dtype=np.float64, copy=True)
        classes = np.array(list(set(y)))

        bns_subscores = np.zeros((len(classes), X.shape[1]))
        for index, target_class in enumerate(classes):
            positive_class_mask = np.array(y == target_class, dtype=int)
            bns_subscores[index, :] = self._generate_bns_subscore(X, positive_class_mask)

        self.ebns_scores = np.max(bns_subscores, axis=0)

        return self

    def transform(self, X):
        if hasattr(X, 'dtype') and np.float64 == np.dtype(X).type:
            X = sp.csr_matrix(X, copy=True)
        else:
            X = sp.csr_matrix(X, dtype=np.float64, copy=True)
        for it, index in enumerate(list(set(X.indices))):
            X.T[index] *= self.ebns_scores[index]

        return sp.coo_matrix(X, dtype=np.float64)

    def _generate_bns_subscore(self, X, class_mask):
        number_positive_docs = np.sum(class_mask)
        number_negative_docs = len(class_mask) - number_positive_docs
        bns_scores = np.ravel(np.zeros((1, X.shape[1])))
        for index, word in enumerate(X.T[:]):
            word_vector = np.ravel(word.toarray())
            bns_scores[index] = self._compute_partial_bns(word_vector, number_positive_docs, number_negative_docs, class_mask)

        return bns_scores

    def _compute_partial_bns(self, word_vector, number_positive_docs, number_negative_docs, class_mask):
        number_true_positive = np.sum(word_vector * class_mask)
        number_false_positive = np.sum(word_vector * np.abs(class_mask - 1))
        true_positive_rate = self.bound_value(float(number_true_positive) / number_positive_docs, self.ppf_limit[0], self.ppf_limit[1])
        false_positive_rate = self.bound_value(float(number_false_positive) / number_negative_docs, self.ppf_limit[0], self.ppf_limit[1])
        bns_score = norm.ppf(true_positive_rate) - norm.ppf(false_positive_rate)

        return bns_score

    @staticmethod
    def bound_value(value, minimum, maximum):
        upper_bounded_value = min(maximum, value)
        upper_and_lower_bounded_value = max(upper_bounded_value, minimum)

        return upper_and_lower_bounded_value
