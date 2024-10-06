from __future__ import annotations
from typing import Callable
from typing import NoReturn
from base_estimator import BaseEstimator
import numpy as np
from numpy.linalg import det, inv

def default_callback(fit: Perceptron, x: np.ndarray, y: int):
    pass


class Perceptron(BaseEstimator):
    """
    Perceptron half-space classifier

    Finds a separating hyperplane for given linearly separable data.

    Attributes
    ----------
    include_intercept: bool, default = True
        Should fitted model include an intercept or not

    max_iter_: int, default = 1000
        Maximum number of passes over training data

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by Perceptron algorithm. To be set in
        `Perceptron.fit` function.

    callback_: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response
    """
    def __init__(self,
                 include_intercept: bool = True,
                 max_iter: int = 1000,
                 callback: Callable[[Perceptron, np.ndarray, int], None] = default_callback):
        """
        Instantiate a Perceptron classifier

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        max_iter: int, default = 1000
            Maximum number of passes over training data

        callback: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.max_iter_ = max_iter
        self.callback_ = callback
        self.coefs_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a halfspace to to given samples. Iterate over given data as long as there exists a sample misclassified
        or that did not reach `self.max_iter_`

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.fit_intercept_`
        """
        n_sample, n_features = np.shape(X)
        self.coefs_ = np.zeros(n_features)
        self.fitted_ = True
        if self.include_intercept_:
            self.coefs_ = np.zeros(n_features + 1)
            ones_column = np.ones((n_sample, 1))
            X = np.hstack((ones_column, X))
            self.coefs_ = np.zeros(n_features + 1)
        for t in range(self.max_iter_):
            exists_i = False
            for i in range(np.shape(X)[0]):
                if y[i] * (self.coefs_ @ X[i]) <= 0:
                    self.coefs_ += y[i] * X[i]
                    exists_i = True
                    break
            if not exists_i:
                break
            self.callback_(self, None, None)


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if self.include_intercept_:
            # Add intercept term to X
            n_samples = np.shape(X)[0]
            ones_column = np.ones((n_samples, 1))
            X = np.hstack((ones_column, X))
        return X @ self.coefs_

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
        from loss_functions import misclassification_error
        y_pred = self.predict(X)
        return misclassification_error(y, y_pred)


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        n_samples, n_features = np.shape(X)

        # Divide samples by their true class
        samples_divided_to_classes = {class_val: X[y == class_val]
                                      for class_val in self.classes_}

        # Calculate pi
        self.pi_ = np.array([len(samples_divided_to_classes[class_val]) / n_samples
                             for class_val in self.classes_])

        # Calculate mu
        self.mu_ = np.array([np.mean(samples, axis=0) for samples in
                             samples_divided_to_classes.values()])

        # Calculate shared covariance matrix
        sum_cov_matrix = np.zeros((n_features, n_features))
        for class_val, samples in samples_divided_to_classes.items():
            centered_samples = samples - self.mu_[self.classes_ == class_val][0]
            sum_cov_matrix += np.dot(centered_samples.T, centered_samples)
        self.cov_ = sum_cov_matrix / n_samples

        # Calculate the inverse of the covariance matrix
        self._cov_inv = np.linalg.inv(self.cov_)
        self.fitted_ = True


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.classes_[np.argmax(self.likelihood(X), axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        likelihoods = np.zeros((n_samples, n_classes))

        # Compute the determinant of the covariance matrix
        det_cov = np.linalg.det(self.cov_)
        const_term = 1 / np.sqrt((2 * np.pi) ** n_features * det_cov)

        # Compute the likelihood for each sample under each class
        for i, class_ in enumerate(self.classes_):
            mu = self.mu_[i]
            pi = self.pi_[i]

            for j, x in enumerate(X):
                # Compute the exponent term
                centered_vector = x - mu
                exponent = -0.5 * np.dot(np.dot(centered_vector, self._cov_inv)
                                         , centered_vector)
                likelihoods[j, i] = const_term * np.exp(exponent) * pi

        return likelihoods


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
        from loss_functions import misclassification_error
        return misclassification_error(y_true=y, y_pred=self._predict(X))


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        n_samples, n_features = np.shape(X)

        # Divide samples by class
        samples_divided_to_classes = {class_val: X[y == class_val]
                                      for class_val in self.classes_}

        # Calculate pi
        self.pi_ = np.array([len(samples_divided_to_classes[class_val]) / n_samples
                             for class_val in self.classes_])

        # Calculate mu
        self.mu_ = np.array([np.mean(samples, axis=0) for samples in
                             samples_divided_to_classes.values()])
        # Calculate vars
        self.vars_ = np.array([np.var(samples, axis=0) for samples in
                             samples_divided_to_classes.values()])
        self.fitted_ = True


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.classes_[np.argmax(self.likelihood(X), axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        likelihoods = np.zeros((n_samples, n_classes))

        for k in range(n_classes):
            # exponent term of the Gaussian PDF
            exponent = np.exp(-(X - self.mu_[k]) ** 2 / (2 * self.vars_[k]))

            # normalization term
            norm_term = 1 / np.sqrt(2 * np.pi * self.vars_[k])

            # product over features
            likelihoods[:, k] = np.prod(exponent * norm_term, axis=1)

        # Multiply by the class priors
        likelihoods *= self.pi_

        return likelihoods

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
        from loss_functions import misclassification_error
        return misclassification_error(y_true=y, y_pred=self._predict(X))
