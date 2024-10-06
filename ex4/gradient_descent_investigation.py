import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from base_module import BaseModule
from base_learning_rate import BaseLR
from cross_validate import cross_validate
from gradient_descent import GradientDescent
from learning_rate import FixedLR
from sklearn.metrics import roc_curve, auc
from loss_functions import misclassification_error
# from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from modules import L1, L2
from logistic_regression import LogisticRegression
from utils import split_train_test, custom

c = [custom[0], custom[-1]]
import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    recorded_values = []
    recorded_weights = []

    def callback(val, weight, **kwargs):
        recorded_values.append(val)
        recorded_weights.append(weight)

    return callback, recorded_values, recorded_weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for name, module in [("L1", L1), ("L2", L2)]:
        results = {}  # To store results for each eta
        for eta in etas:
            callback, values, weights = get_gd_state_recorder_callback()
            gd = GradientDescent(learning_rate=FixedLR(eta), callback=callback)
            gd.fit(module(weights=init.copy()), None, None)

            # Question 1: Plot the descent path
            descent_path = np.array(weights)
            fig = plot_descent_path(module, descent_path, title=f"{name} with Learning Rate: {eta}")
            fig.write_html(f"{name}_{eta}_descent.html")

            # Save convergence rate
            results[eta] = values

        # Question 3: Plot the convergence rate
        fig_conv = go.Figure()
        for eta, values in results.items():
            fig_conv.add_trace(go.Scatter(x=list(range(len(values))), y=values, mode="lines", name=f"η={eta}"))

        fig_conv.update_layout(
            xaxis_title="GD Iteration",
            yaxis_title="Norm",
            title=f"{name} GD Convergence For Different Learning Rates"
        )
        fig_conv.write_html(f"gd_{name}_fixed_rate_convergence.html")

        # Question 4: Print the lowest loss achieved for each eta
        best_eta = min(results, key=lambda eta: np.min(results[eta]))
        best_loss = np.min(results[best_eta])
        print(f"The lowest loss achieved by a {name} module with a fixed learning rate η={best_eta} "
              f"is {np.round(best_loss, 10)}")



def load_data(path: str = "SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data

    callback, _, _ = get_gd_state_recorder_callback()
    gd = GradientDescent(callback=callback)

    # Initialize and fit the logistic regression model
    lr_model = LogisticRegression(solver=gd)
    lr_model.fit(X_train.values, y_train.values)

    # Predict probabilities for the test set
    y_proba = lr_model.predict_proba(X_test.values)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=f"ROC Curve Of Fitted Model - AUC={auc(fpr, tpr):.6f}",
                         xaxis=dict(title="False Positive Rate (FPR)"),
                         yaxis=dict(title="True Positive Rate (TPR)"))).write_html("roc_curve.html")

    # Model loss on test for threshold maximizing TPR-FPR
    lr_model.alpha_ = thresholds[np.argmax(tpr - fpr)]
    print("Best alpha for (TPR - FPR) ratio is:", str(np.round(lr_model.alpha_, decimals=5)),
          f"\nThe loss on the test with such alpha is: ",
          np.round(lr_model.loss(X_test.values, y_test.values), decimals=5))

    # Fitting l1- logistic regression model, using cross-validation to specify values
    # of regularization parameter
    lams_options = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    validation_scores = []
    train_scores = []
    lr = 1e-4
    max_iter = 20000
    for lam in lams_options:
        gd = GradientDescent(learning_rate=FixedLR(lr), max_iter=max_iter)
        lr_reg = LogisticRegression(solver=gd, penalty="l1", lam=lam, alpha=.5)
        train_score, validation_score = cross_validate(lr_reg,
                                                       X_train.values,
                                                       y_train.values,
                                                       scoring=misclassification_error)
        validation_scores.append(validation_score)
        train_scores.append(train_score)

    # Selecting optimal lambda
    fig = go.Figure([go.Scatter(x=lams_options, y=train_scores, name="Train Error"),
                     go.Scatter(x=lams_options, y=validation_scores, name="Validation Error")],
                    layout=go.Layout(
                        title="Train and Validation Errors As Functions of lambda",
                        xaxis=dict(title="lambda", type="log"),
                        yaxis=dict(title="Error Value")))
    fig.write_html("logistic_cross_validation_errors.html")

    validation_scores = np.array(validation_scores)
    best_lam = lams_options[int(np.argmin(validation_scores))]
    lr_model = LogisticRegression(solver=gd, penalty="l1", lam=best_lam).fit(X_train.values, y_train.values)
    print("l1 module:")
    print(f"Best lambda:{best_lam}")
    print(f"Model achieved test error of:"
          f" {np.round(lr_model.loss(X_test.values, y_test.values), decimals=2)}")

if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    fit_logistic_regression()
