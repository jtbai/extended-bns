## This is greatly inspired from the implementation of Mathieu Dumoulin
## https://github.com/dumoulma/fic-prototype

from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.sparse as sp
import numpy as np


class EBNSTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, ppf_limit=(0.0005, 1 - 0.0005)):
        self.ppf_limit = ppf_limit
        self.bns_scores = []
        self.scoring_function = lambda x: np.max(x, axis=0)
        self.bns_score_nb_positives = []

    def fit(self, X, y):
        y = np.array(y)
        if hasattr(X, 'dtype') and np.float64 == np.dtype(X).type:
            X = sp.csr_matrix(X, copy=True)
        else:
            X = sp.csr_matrix(X, dtype=np.float64, copy=True)
        classes = np.array(list(set(y)))
        self.bns_scores = np.zeros((len(classes), X.shape[1]))
        for index, target_class in enumerate(classes):
            class_mask = np.array(y == target_class, dtype=int)
            self.bns_scores[index, :] = self._generate_bns_score(X, class_mask)
        self.bns_score_nb_positives = np.sum(self.bns_scores > 0, axis=0)
        self.bns_scores = np.abs(self.scoring_function(self.bns_scores))

        return self

    def transform(self, X):
        if hasattr(X, 'dtype') and np.float64 == np.dtype(X).type:
            X = sp.csr_matrix(X, copy=True)
        else:
            X = sp.csr_matrix(X, dtype=np.float64, copy=True)
        for it, index in enumerate(list(set(X.indices))):
            X.T[index] *= self.bns_scores[index]

        return sp.coo_matrix(X, dtype=np.float64)

    def _generate_bns_score(self, X, class_mask):
        positive_doc = np.sum(class_mask)
        negative_doc = len(class_mask) - positive_doc
        bns_scores = np.ravel(np.zeros((1, X.shape[1])))
        for index, word in enumerate(X.T[:]):
            word_vector = np.ravel(word.toarray())
            bns_scores[index] = self._compute_partial_bns(word_vector, positive_doc, negative_doc, class_mask)

        return bns_scores

    def _compute_partial_bns(self, word_vector, pos, neg, class_mask):
        tp = np.sum(word_vector * class_mask)
        fp = np.sum(word_vector * np.abs(class_mask - 1))
        tpr = min(self.ppf_limit[1], max(self.ppf_limit[0], float(tp) / pos))
        fpr = min(self.ppf_limit[1], max(self.ppf_limit[0], float(fp) / neg))
        bns_score = norm.ppf(tpr) - norm.ppf(fpr)

        return bns_score
