import pandas as pd
import datetime
import pickle
import matplotlib
import numpy as np

from basic_analysis_toolkit import BasicAnalysis
# each portfolio has two objects: a overview dataframe and a historical dataframe
# the overview dataframe has the indexes as the list of tickers and the columns as
# initial value,

# the information will be inputted as a
# each portfolio has two objects: a overview dataframe and a historical dataframe
# the overview dataframe has the indexes as the list of tickers and the columns as
# initial value,

# the information will be inputted as a dictionary of invesments. Example:
# d = {'AAPL': {'Date': xx/xx/xxxx, 'No of Shares': XX, 'Initial Market Value': 'XX'}}
# because the user gets to manually input the information, it is up to them to make sure
# it is formatted correctly and the information is accurate


class Portfolio:
    def __init__(self, dict_of_investments):
        self.overview = pd.DataFrame(dict_of_investments)

    def get_historical_performance(self, analysis, end_date):
        list_initial_dates = []
        dict_performances = {}
        for ticker in self.overview.columns:
            num_shares = self.overview.get(ticker).get('No of Shares')
            initial_date = self.overview.get(ticker).get('Date')
            list_initial_dates.append(initial_date)
            idx = pd.date_range(initial_date, end_date, freq='B')
            values = analysis.get_close(ticker, initial_date, end_date, 'B') * num_shares
            dict_performances.update({ticker: values})
        oldest_date = max(list_initial_dates)
        df_idx = pd.date_range(oldest_date, end_date, freq='B')
        historical_performance = pd.DataFrame(dict_performances, index=df_idx).fillna(method='pad')
        historical_performance['Total Value'] = historical_performance.sum(axis=1, skipna=True)
        return historical_performance

    # getting a graph of the performance of each investment in terms of return
    def get_return_df(self, analysis, end_date):
        # replace with self method
        df = self.get_historical_performance(analysis, end_date)
        new_df = pd.DataFrame()
        initial_date = df.index[-1]
        for ticker in list(df.columns):
            initial_costs = df[ticker].iloc[0]
            new_df[ticker] = df[ticker] / initial_costs - 1
        return new_df

# given the matrix of covariances, this method returns a vector of weights that represents the optimal portfolio
# optimized to provide the minimum variance
def min_var_opt(sigma):
    n = len(sigma.columns)
    i = np.ones(n)
    sigma_inverse = pd.DataFrame(np.linalg.pinv(sigma.values), sigma.columns, sigma.index)
    numerator = sigma_inverse.dot(i)
    denominator = i.T.dot(sigma_inverse).dot(i)
    return numerator/denominator


#calculates the variance of the portfolio given a series that contains the weighting information of the portfolio and
#a matrix of covariances
def calc_var(sigma, w):
    return w.T.dot(sigma).dot(w)


# calculates the optimal weights given a risk_parameter, forecasts, covariances, and whether the portfolio includes
# a cash portion or not
def mean_var_opt(risk_param, f_e, sigma, cash):
    sigma_inverse = pd.DataFrame(np.linalg.pinv(sigma.values), sigma.columns, sigma.index)
    if cash:
        weights = (1 / risk_param) * sigma_inverse.dot(f_e)
        cash_weight = 1 - weights.T.dot(np.ones(len(weights)))
        return {'Equity Weights': weights, 'Cash Weight': cash_weight}
    else:
        n = len(sigma.columns)
        i = np.ones(n)
        numer1 = sigma_inverse.dot(i)
        denom1 = i.T.dot(sigma_inverse).dot(i)
        min_var_sol = numer1 / denom1
        numer2 = (i.T.dot(sigma_inverse).dot(i)) * sigma_inverse.dot(f_e) - (
            i.T.dot(sigma_inverse).dot(f_e)) * sigma_inverse.dot(i)
        return min_var_sol + (1 / risk_param) * (numer2) / (denom1)


# given a general an attribute, this computes the minimum variance portfolio with unit exposure to attribute t e
def characteristic_portfolio_weights(sigma, t):
    n = len(sigma.columns)
    i = np.ones(n)
    sigma_inverse = pd.DataFrame(np.linalg.pinv(sigma.values), sigma.columns, sigma.index)
    numer = sigma_inverse.dot(t)
    denom = t.T.dot(sigma_inverse).dot(t)
    return numer / denom
