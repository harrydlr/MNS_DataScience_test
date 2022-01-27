import datetime
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

def fill_missing_dates(data, update_date):
    idx = pd.date_range(data["date"].min(), pd.Timestamp(update_date), freq="168h")
    data = data.set_index(['date'])
    data = data.reindex(idx, fill_value=-0.0).reset_index()
    data = data.rename(columns={'index': 'date'})
    return data

def data_forecasting(data):
    weeks_in_month = 4
    differenced = difference(data, weeks_in_month)
    # fit model
    model = ARIMA(differenced, order=(7, 1, 2))
    model_fit = model.fit()
    # multi-step out-of-sample forecast
    forecast = model_fit.forecast(steps=4)
    # invert the differenced forecast to something usable
    history = [x for x in data]
    week = 1
    predictions = []
    for yhat in forecast:
        inverted = inverse_difference(history, yhat, weeks_in_month)
        predictions.append(inverted)
        history.append(inverted)
        week += 1

    return predictions

def forecast_model(transactions, up_date):
    transactions = pd.DataFrame(map(dict, transactions))
    transactions.date = transactions.date.apply(lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d'))
    transactions = transactions.groupby(["date"])['amount'].agg(
        [('negative_trans', lambda x: x[x < 0].sum()), ('positive_trans', lambda x: x[x > 0].sum())]).reset_index()

    negative_trs = transactions[["date", "negative_trans"]]
    weekly_negative_trs = negative_trs.groupby([pd.Grouper(key='date', freq='168h')])[
        "negative_trans"].sum().reset_index()
    final_df = fill_missing_dates(weekly_negative_trs, up_date)
    print(final_df)
    final_df.loc[final_df['negative_trans'] == 0.00, 'negative_trans'] = -0.00
    X = final_df.set_index(['date']).negative_trans.sort_index().values
    forecasting_results = data_forecasting(X)
    month_outgoing = sum(forecasting_results)

    return month_outgoing

