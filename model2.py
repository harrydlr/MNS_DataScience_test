import datetime
from statsmodels.tsa.vector_ar.var_model import VAR
import pandas as pd


def fill_missing_dates(data, update_date):
    idx = pd.date_range(data["date"].min(), pd.Timestamp(update_date), freq="168h")
    data = data.set_index(['date'])
    data = data.reindex(idx, fill_value=-0.0).reset_index()
    data = data.rename(columns={'index': 'date'})
    return data

def data_forecasting(data):
    # fit the model
    model = VAR(endog=data)
    model_fit = model.fit()

    # make prediction on validation
    prediction = model_fit.forecast(model_fit.model.y, steps=4)

    res = [x[0] for x in prediction]

    return res

def forecast_model(transactions, up_date, balance):
    transactions = pd.DataFrame(map(dict, transactions))
    transactions.date = transactions.date.apply(lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d'))
    transactions = transactions.groupby(["date"])['amount'].agg(
        [('negative_trans', lambda x: x[x < 0].sum()), ('positive_trans', lambda x: x[x > 0].sum())]).reset_index()

    weekly_valid_transactions_df = transactions.groupby([pd.Grouper(key='date', freq='168h')])[
        ["negative_trans", "positive_trans"]].sum().reset_index()

    weekly_valid_transactions_df = fill_missing_dates(weekly_valid_transactions_df, up_date)
    weekly_valid_transactions_df.loc[weekly_valid_transactions_df['negative_trans'] == 0.00, 'negative_trans'] = -0.00
    weekly_valid_transactions_df.loc[weekly_valid_transactions_df['positive_trans'] == -0.00, 'positive_trans'] = 0.00

    weekly_valid_transactions_df["total_trans"] = weekly_valid_transactions_df["negative_trans"] + \
                                                  weekly_valid_transactions_df["positive_trans"]

    # Compute intial balance
    initial_balance = round(balance - weekly_valid_transactions_df["total_trans"].sum(), 2)
    # Compute balance of each entry in weekly_valid_transactions_df
    weekly_valid_transactions_df['balance'] = 0
    aux_df = weekly_valid_transactions_df.copy()
    for group_name, group_df in aux_df.groupby(["date"]):
        date = group_name
        if date == (weekly_valid_transactions_df["date"].min()):
            weekly_valid_transactions_df.loc[(weekly_valid_transactions_df.date == date), 'balance'] = initial_balance
        else:
            previous_data = weekly_valid_transactions_df[(weekly_valid_transactions_df["date"] < date)]
            weekly_valid_transactions_df.loc[(weekly_valid_transactions_df.date == date), 'balance'] = previous_data[
                                                                                                           "total_trans"].sum() + initial_balance

    final_df = weekly_valid_transactions_df[["date", "negative_trans", "balance", "positive_trans"]]
    final_df = final_df.set_index(final_df['date'])
    final_df = final_df.drop(['date'], axis=1)

    forecasting_results = data_forecasting(final_df)
    month_outgoing = sum(forecasting_results)

    return month_outgoing
