from classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loss_functions import accuracy
from math import atan2, pi
import numpy as np


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)
        X = np.array(X)
        y = np.array(y)
        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def find_loss(fit: Perceptron, _, __):
            losses.append(fit.loss(X, y))

        Perceptron(callback=find_loss).fit(X, y)
        # Plot figure of loss as function of fitting iteration
        fig = go.Figure(data=go.Scatter(x=list(range(len(losses))),
                                        y=losses,
                                        mode="lines",
                                        marker=dict(color="navy")))
        fig.update_layout(
            title="Perceptron Training Error - {} dataset".format(n),
            xaxis_title="Iteration",
            yaxis_title="Misclassification Error"
        )

        fig.write_html(f"perceptron_fit_{n}.html")


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """

    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """

    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)
        X = np.array(X)
        y = np.array(y)
        # Fit models and predict over training set
        naive = GaussianNaiveBayes().fit(X, y)
        naive_preds = naive.predict(X)
        lda = LDA().fit(X, y)
        lda_preds = lda.predict(X)
        gaussian_accuracy_percentage = round(100 * accuracy(y, naive_preds), 2)
        lda_accuracy_percentage = round(100 * accuracy(y, lda_preds), 2)
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes
        # predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot
        # titles should specify algorithm and accuracy
        # Create subplots
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(
                                f"Gaussian Naive Bayes (accuracy="
                                f"{gaussian_accuracy_percentage}%)",
                                f"LDA (accuracy="
                                f"{lda_accuracy_percentage}%)"))

        fig.update_layout(title_text=f"Comparing Gaussian Classifiers"
                                     f" - {f[:-4]} dataset",
                          width=800, height=400, showlegend=False)

        # Add traces for data-points setting symbols and colors of GNB

        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                 marker=dict(color=naive_preds,
                                             symbol=class_symbols[y],
                                             colorscale=class_colors
                                             (len(np.unique(y))))),
                      row=1, col=1)

        # Add traces for data-points setting symbols and colors of LDA

        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                 marker=dict(color=lda_preds,
                                             symbol=class_symbols[y],
                                             colorscale=class_colors(len(
                                                 np.unique(y))))),
                      row=1, col=2)

        # Add markers representing the means of the fitted Gaussian
        # distribution
        fig.add_trace(go.Scatter(x=naive.mu_[:, 0],
                                 y=naive.mu_[:, 1],
                                 mode="markers",
                                 marker=dict(symbol="x",
                                             color="black",
                                             size=16)),
                      row=1, col=1)

        # Add markers representing the means of the fitted LDA distribution
        fig.add_trace(go.Scatter(x=lda.mu_[:, 0],
                                 y=lda.mu_[:, 1],
                                 mode="markers",
                                 marker=dict(symbol="x",
                                             color="black",
                                             size=16)),
                      row=1, col=2)

        # Add ellipses representing the covariances of the fitted
        # Gaussian distributions and LDA

        for i in range(len(np.unique(y))):
            fig.add_traces([get_ellipse(naive.mu_[i], np.diag(naive.vars_[i])),
                            get_ellipse(lda.mu_[i], lda.cov_)],
                           rows=[1, 1], cols=[1, 2])

        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.write_html(f"lda.and.naive.bayes.{f[:-4]}.html")


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
