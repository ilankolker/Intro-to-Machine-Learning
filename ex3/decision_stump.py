from __future__ import annotations

import math
from typing import Tuple, NoReturn
from base_estimator import BaseEstimator
import numpy as np
from itertools import product
from loss_functions import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        error = np.inf
        for feature in range(np.shape(X)[1]):
            for sign in [-1, 1]:
                thr, thr_err = self._find_threshold(X[:, feature], y, sign)
                if thr_err < error:
                    error = thr_err
                    self.j_ = feature
                    self.threshold_ = thr
                    self.sign_ = sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sign responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        y_predict = np.where(X[:, self.j_] < self.threshold_, -self.sign_, self.sign_)
        return y_predict

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # sort values and labels respectively, used for calculating
        # loss efficiently using cumsum
        sorted_indices = np.argsort(values)
        values = values[sorted_indices]
        labels = labels[sorted_indices]

        # Loss for classifying all as sign (when threshold < values[0])
        # This loss can include only false positive if sign = 1 or
        # false negative if sign = -1
        loss_threshold_minus_inf = np.sum(np.abs(labels)[np.sign(labels) != sign])

        # Loss of classifying threshold of each value
        cumulative_losses = loss_threshold_minus_inf - np.cumsum(labels * -sign)

        loss_vec = np.zeros(len(cumulative_losses) + 1)
        loss_vec[0] = loss_threshold_minus_inf
        loss_vec[1:] = cumulative_losses

        # Find the index of the minimal loss
        index_minimal_loss = np.argmin(loss_vec)
        if index_minimal_loss == 0:
            threshold = -np.inf
            min_loss = loss_threshold_minus_inf
        elif index_minimal_loss == np.shape(loss_vec)[0] - 1:
            threshold = np.inf
            min_loss = loss_vec[index_minimal_loss]
        else:
            threshold = values[index_minimal_loss]
            min_loss = loss_vec[index_minimal_loss]

        return threshold, float(min_loss)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.predict(X))
