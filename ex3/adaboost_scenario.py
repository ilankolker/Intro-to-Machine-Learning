import numpy as np
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adaboost import AdaBoost
from decision_stump import DecisionStump


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    train_X, train_y = generate_data(train_size, noise)
    test_X, test_y = generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    model = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    train_error = []
    test_error = []
    # Get all errors from train set as function of t (number of learners)
    for t in range(1, n_learners + 1):
        train_error.append(model.partial_loss(train_X, train_y, t))

    # Get all errors from test set as function of t (number of learners)
    for t in range(1, n_learners + 1):
        test_error.append(model.partial_loss(test_X, test_y, t))

    # Plotting the results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, n_learners + 1)),
                             y=train_error, mode='lines', name='Train Error'))
    fig.add_trace(go.Scatter(x=list(range(1, n_learners + 1)), y=test_error,
                             mode='lines', name='Test Error'))
    fig.update_layout(
        title=dict(text="AdaBoost Misclassification As Function Of Number of Learners",
                   font=dict(size=14)),
        xaxis_title="Iteration",
        yaxis_title="Misclassification Error",
        template="simple_white",
        width=500,
        height=500
    )
    fig.write_html(f"adaboost_{noise}.html")

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    # Create subplots with titles for each iteration in T
    fig_decision_boundaries = make_subplots(rows=1, cols=len(T), subplot_titles=[f"{t} Classifiers" for t in T],
                                            horizontal_spacing=0.05, vertical_spacing=0.1)

    for idx, t in enumerate(T):
        # Get predictions up to iteration t
        partial_predict_func = lambda X: model.partial_predict(X, t)

        # Decision surface for current iteration t
        decision_surface_trace = decision_surface(partial_predict_func,
                                                  lims[0],
                                                  lims[1],
                                                  density=60,
                                                  showscale=False)

        # Scatter plot of the test set
        test_scatter_trace = go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                                        showlegend=False,
                                        marker=dict(color=test_y,
                                                    symbol=np.where(test_y == 1, "circle", "x"),
                                                    line=dict(color="black", width=1),
                                                    colorscale=custom, showscale=False),
                                        name='Test Set')

        # Add traces to the subplots
        fig_decision_boundaries.add_trace(decision_surface_trace, row=1, col=idx + 1)
        fig_decision_boundaries.add_trace(test_scatter_trace, row=1, col=idx + 1)

    fig_decision_boundaries.update_layout(
        width=1500,
        height=500,
        margin=dict(t=100),
        template="simple_white"
    ).update_xaxes(visible=False).update_yaxes(visible=False)
    fig_decision_boundaries.write_html(f"adaboost_{noise}_decision_boundaries.html")

    # Question 3: Decision surface of best performing ensemble

    # Finding the best ensemble size based on test error
    best_t = np.argmin(test_error) + 1  # +1 because t is 1-indexed
    best_accuracy = 1 - test_error[best_t - 1]

    # Plotting the decision surface of the best ensemble
    fig_best_ensemble = go.Figure()

    decision_surface_trace_best = decision_surface(lambda X: model.partial_predict(X, best_t), lims[0], lims[1],
                                                   density=60, showscale=False)

    # Scatter plot of the test set
    test_scatter_trace_best = go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                                         marker=dict(color=test_y, symbol=np.where(test_y == 1, "circle", "x"),
                                                     line=dict(color="black", width=1),
                                                     colorscale=custom, showscale=False),
                                         name='Test Set')

    fig_best_ensemble.add_trace(decision_surface_trace_best)
    fig_best_ensemble.add_trace(test_scatter_trace_best)
    fig_best_ensemble.update_layout(
        title=dict(text=f"Decision Boundary of Best Ensemble (Size: {best_t}, Accuracy: {best_accuracy:.2f})",
                   font=dict(size=14)),
        width=600,
        height=600,
        template="simple_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    fig_best_ensemble.write_html(f"adaboost_{noise}_best_ensemble.html")

    # Question 4: Decision surface with weighted samples
    # Normalizing and transforming weights for plotting
    D = model.D_
    D_normalized = D / np.max(D) * 5

    fig_weights = go.Figure()

    # Decision surface using the full ensemble
    decision_surface_trace_full = decision_surface(model.predict, lims[0], lims[1],
                                                   density=60, showscale=False)

    # Scatter plot of the training set with normalized weights
    train_scatter_trace = go.Scatter(x=train_X[:, 0],
                                     y=train_X[:, 1],
                                     mode="markers",
                                     showlegend=False,
                                     marker=dict(size=D_normalized, color=train_y,
                                                 symbol=np.where(train_y == 1, "circle", "x"),
                                                 line=dict(color="black", width=1),
                                                 colorscale=custom, showscale=False),
                                     name='Training Set')

    fig_weights.add_trace(decision_surface_trace_full)
    fig_weights.add_trace(train_scatter_trace)
    fig_weights.update_layout(
        title=dict(text=f"Final AdaBoost Sample Distribution",
                   font=dict(size=16)),
        width=600,
        height=600,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    fig_weights.write_html(f"adaboost_{noise}_training_weights.html")


if __name__ == '__main__':
    np.random.seed(0)
    for noise in [0, 0.4]:
        fit_and_evaluate_adaboost(noise)
