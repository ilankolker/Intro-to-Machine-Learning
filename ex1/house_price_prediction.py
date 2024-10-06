from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import linear_regression


def fix_X_nan_values(X):
    """
    Replace NaN values in X with the mean of their respective columns and
    delete rows with none values in price (in y, and X respectively).
    Round the 'date' column in X after replacing NaNs.
    Returns:
    - X (DataFrame): Feature matrix with NaNs replaced by column means and
    deleted rows correspond to the nan prices in y.
    - y (Series): Target vector without nan values.
    """
    mean_X_values = X.mean()
    X = X.fillna(mean_X_values)
    X['date'] = X['date'].apply(lambda x: round(x))
    return X


def remove_nan_y_values(X, y):
    """
    Remove rows with NaN values in the target (y) and corresponding rows
     in the features (X).
    Returns:
    - X (DataFrame): The feature matrix with rows removed
     where y has NaN values.
    - y (Series): The target vector with NaN values removed.
    """
    nan_indices_y = y.index[y.isna()]
    X = X.drop(nan_indices_y)
    y = y.dropna()
    return X, y


def replace_invalid_X_vals_with_mean(X):
    """
    Replace invalid or negative values in features (X) with the mean
     of their respective columns.
    Parameters:
    - X (DataFrame): The feature matrix.
    Returns:
    - X (DataFrame): The feature matrix with negative values replaced by
     column means.
    """
    validity_checks = {
        'sqft_lot': (X['sqft_lot'] > 0),
        'sqft_living': (X['sqft_living'] > 0),
        'view': X['view'].isin(range(5)),
        'yr_built': (X['yr_built'] > 0),
        'yr_renovated': (X['yr_renovated'] >= 0),
        'sqft_above': (X['sqft_above'] > 0),
        'bathrooms': (X['bathrooms'] >= 0),
        'floors': (X['floors'] >= 0),
        'waterfront': X['waterfront'].isin([0, 1]),
        'condition': X['condition'].isin(range(1, 6)),
        'sqft_basement': (X['sqft_basement'] >= 0),
        'grade': X['grade'].isin(range(1, 14))
    }
    for col, condition in validity_checks.items():
        mean_value = int(X.loc[condition, col].mean())
        X.loc[~condition, col] = mean_value

    return X


def remove_non_positive_y(X, y):
    """
    Remove rows in X and y where y has non-positive values.
    Returns:
    - X (DataFrame): The feature matrix with rows removed where y
     had non-positive values.
    - y (Series): The target vector with non-positive values removed.
    """
    # Find indices of rows with non-positive values in y
    non_positive_indices = y.index[y <= 0]
    X = X.drop(non_positive_indices)
    y = y.drop(non_positive_indices)
    return X, y


def replace_invalid_year_renovated(X):
    """
    Replace invalid values in the 'yr_renovated' column with 0 if the year
     renovated is before the year built.
    Parameters:
    - X (DataFrame): The feature matrix.
    Returns:
    - X (DataFrame): The feature matrix with invalid 'yr_renovated'
     values replaced by 0.
    """
    mask_x = (X['yr_renovated'] != 0) & (X['yr_renovated'] < X['yr_built'])
    X.loc[mask_x, 'yr_renovated'] = 0
    return X


def categorize_renovation(X):
    """
    Categorizes houses based on their renovation status.
    Returns:
    X: A DataFrame with an additional column 'recently_renovated'
            containing the renovation status category for each house.
    Categories:
    - 3: Houses renovated more than 0 years but less than or equal to
     10 years ago.
    - 2: Houses renovated more than 10 years but less than or
     equal to 30 years since .
    - 1: Houses renovated more than 30 years ago.
    - 0: Houses not renovated at all.
    If a house has a renovation year of 0, it's categorized as 0
    """
    years_since_renovation = abs(X['date'] - X['yr_renovated'])

    # Define the bins and labels for categorization
    bins = [0, 10, 30, float('inf')]
    labels = [3, 2, 1]  # Corresponding categories
    X.loc[X['yr_renovated'] == 0, 'recently_renovated'] = 0
    X.loc[X['yr_renovated'] != 0, 'recently_renovated'] = pd.cut(
        years_since_renovation,
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True
    ).astype(int)
    return X


def add_bathrooms_per_bedrooms_ratio(X):
    """
    Add a new feature representing the ratio of bathrooms to
     bedrooms in the dataset.
    Parameters:
    - X (DataFrame): The feature matrix.
    Returns:
    - X (DataFrame): The feature matrix with the new feature
     'bathrooms_per_bedrooms_ratio' added.
    """
    X['bathrooms_per_bedrooms_ratio'] = X['bathrooms'] / X['bedrooms'] \
        .where(X['bedrooms'] != 0)
    X.loc[X['bedrooms'] == 0, 'bathrooms_per_bedrooms_ratio'] = 0
    return X


def drop_column(X, column_name):
    """
    Drop a specified column from the feature matrix.
    Returns:
    - X (DataFrame): The feature matrix with the specified column dropped.
    """
    X = X.drop([column_name], axis=1)
    return X


def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """
    # drops the id column from X
    X = drop_column(X, 'id')
    # formats the date in X matrix to be a float representing Year
    X['date'] = pd.to_datetime(X['date'])
    X['date'] = pd.DatetimeIndex(X['date']).year
    # filters rows that has NAN value in X and y
    X = fix_X_nan_values(X)
    X, y = remove_nan_y_values(X, y)
    X = replace_invalid_X_vals_with_mean(X)
    X, y = remove_non_positive_y(X, y)
    X = replace_invalid_year_renovated(X)
    # categorize renovations based on how recently houses were renovated
    X = categorize_renovation(X)
    X = drop_column(X, 'date')
    X = add_bathrooms_per_bedrooms_ratio(X)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return X, y


def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    X = drop_column(X, 'id')
    X['date'] = pd.to_datetime(X['date'])
    X['date'] = pd.DatetimeIndex(X['date']).year
    X = fix_X_nan_values(X)
    X = replace_invalid_X_vals_with_mean(X)
    X = replace_invalid_year_renovated(X)
    X = categorize_renovation(X)
    X = drop_column(X, 'date')
    X = add_bathrooms_per_bedrooms_ratio(X)
    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") \
        -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # Iterate over each feature
    for feature_name in X.columns:
        # Compute covariance between feature and response
        cov_xy = np.cov(X[feature_name], y)[0, 1]
        # Compute standard deviations of feature and response
        std_x = np.std(X[feature_name])
        std_y = np.std(y)
        # Compute Pearson Correlation
        correlation = cov_xy / (std_x * std_y)
        img = px.scatter(x=X[feature_name],
                         y=y,
                         title=f"{feature_name} - Pearson Correlation:"
                               f" {correlation:.2f}",
                         labels={"x": feature_name, "y": "Price"})
        img.write_html(f"{output_path}/{feature_name}_scatter.html")


if __name__ == '__main__':
    # Question 2 - split train test
    df = pd.read_csv("house_prices.csv")
    random_seed = 50
    shuffled_df = df.sample(frac=1, random_state=random_seed)
    X, y = df.drop("price", axis=1), df.price
    split_index = int(0.75 * len(shuffled_df))
    # Split the DataFrame into training and test sets
    train_set = X.iloc[:split_index]
    test_set = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    # Question 3 - preprocessing of housing prices train dataset
    train_pre_processed, y_pre_processed_train =\
        preprocess_train(train_set, y_train)
    # Question 4 - Feature evaluation of train dataset with respect to response
    feature_evaluation(train_pre_processed, y_pre_processed_train)
    # Question 5 - preprocess the test data
    test_set_pre_processed = preprocess_test(test_set)
    # Question 6 - Fit model over increasing percentages of the
    # overall training data
    model = linear_regression.LinearRegression()
    percentages = list(range(10, 101))
    mean_losses = []
    lower_bounds = []
    upper_bounds = []
    for p in range(10, 101):
        losses = []
        for i in range(10):
            # For every percentage p in 10%, 11%, ..., 100%, repeat
            # the following 10 times 1) Sample p% of the overall training data
            sampled_indices = train_pre_processed.sample(frac=p / 100).index
            sampled_train_set = train_pre_processed.loc[sampled_indices]
            sampled_y_train = y_pre_processed_train.loc[sampled_indices]
            # 2) Fit linear model (including intercept) over sampled set
            model.fit(sampled_train_set, sampled_y_train)
            # 3) Test fitted model over test set
            loss = model.loss(test_set_pre_processed, y_test)
            losses.append(loss)
        # 4) Store average and variance of loss over test set
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        lower_bound = mean_loss - 2 * std_loss
        upper_bound = mean_loss + 2 * std_loss
        mean_losses.append(mean_loss)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

    mean_trace = go.Scatter(
        x=percentages,
        y=mean_losses,
        mode='lines+markers',
        name='Mean Loss'
    )

    confidence_interval_trace = go.Scatter(
        x=percentages + percentages[::-1],
        y=lower_bounds + upper_bounds[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='Confidence Interval'
    )
    # Create layout
    layout = go.Layout(
        title='Mean Loss as Function of Sampling Percentage',
        xaxis=dict(title='Percentage of Data Sampled (%)'),
        yaxis=dict(title='Mean Loss'),
        template='plotly_white'
    )
    # Create figure
    go.Figure(data=[mean_trace, confidence_interval_trace],
              layout=layout).write_html("./mse.over.data.percentage.html")
