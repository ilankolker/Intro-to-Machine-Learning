import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
import polynomial_fitting


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset
    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date']).dropna().drop_duplicates()
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df = df[df.Temp > 0]
    df = df[(df.Month <= 12) & (df.Month >= 1)]
    df = df[(1 <= df.Day) & (df.Day <= 31)]
    return df


if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")

    # Question 3 - Exploring data for specific country
    df_filtered_israel = df[df.Country == "Israel"]
    px.scatter(df_filtered_israel,
               x="DayOfYear",
               y="Temp",
               color=df_filtered_israel['Date'].dt.year.astype(str),
               title="Temperature by Days Of Year") \
        .write_html("./israel_daily_temp.html")

    px.bar(df_filtered_israel.groupby('Month', as_index=False).agg
           (std=("Temp", "std")), x="Month",
           y="std", title="Temperature Standard Deviation Over Months"). \
        write_html("./israeli_monthly_average_temp.html")

    # Question 4 - Exploring differences between countries
    # Group the samples by 'Country' and 'Month' and calculate the
    # average and standard deviation of the temperature
    grouped_data = df.groupby(['Country', 'Month'])['Temp'].agg \
        (avg_temp='mean', std_temp='std').reset_index()
    # Plot the line plot with error bars
    px.line(grouped_data,
            x='Month',
            y='avg_temp',
            labels={'Month': 'Month', 'avg_temp': 'Average Temperature'},
            error_y='std_temp',
            color='Country',  # Color-coded by country
            title='Average Monthly Temperature by Country') \
        .write_html("./mean.temp.different.countries.html")

    # Question 5 - Fitting model for different values of `k`
    shuffled_df = df_filtered_israel.sample(frac=1)
    X, y = shuffled_df, shuffled_df
    train_set, test_set, y_train, y_test = train_test_split \
        (X, y, test_size=0.25)
    losses = []
    for k in range(1, 11):
        polynomial_model = polynomial_fitting.PolynomialFitting(k)
        polynomial_model.fit(train_set.DayOfYear, y_train.Temp)
        loss = np.round(polynomial_model.loss
                        (test_set.DayOfYear, y_test.Temp), 2)
        losses.append(loss)

    k_values = range(1, 11)
    # pair each element of k_values with its corresponding element from losses
    for k, loss in zip(k_values, losses):
        print(f'Test error for k={k}: {loss}')

    px.bar(
        x=k_values,
        y=losses,
        text=[f'{loss:.2f}' for loss in losses],
        labels={'x': 'k', 'y': 'Test loss'},
        title=r"Test loss For Different Values of k") \
        .write_html("./israel.different.k.html")

    # Question 6 - Evaluating fitted model on different countries
    model = polynomial_fitting.PolynomialFitting(5)
    model.fit(df_filtered_israel.DayOfYear, df_filtered_israel.Temp)
    countries = ["Jordan", "South Africa", "The Netherlands"]
    losses = []
    # Loop through each country and calculate the loss
    for country in countries:
        country_data = df[df['Country'] == country]
        loss = round(model.loss(country_data['DayOfYear'],
                                country_data['Temp']), 2)
        losses.append({"Country": country, "Loss": loss})

    losses_df = pd.DataFrame(losses)
    px.bar(
        losses_df,
        x="Country",
        y="Loss",
        text="Loss",
        color="Country",
        title="Loss Over Countries For Model Fitted Over Israel"
    ).write_html("./test.other.countries.html")
