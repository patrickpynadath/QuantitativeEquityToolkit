import DataRetrieval
import pandas as pd
import datetime
from sklearn import linear_model
from DataRetrieval import get_avail_tickers, update_dataframe_ticker
import pickle
import pandas_datareader as pdr
from company_toolkit import *
import numpy as np
import math

class BasicAnalysis:
    def __init__(self, df):
        self.df = df

    def get_close(self, ticker, start, end, f):
        if ticker in get_avail_tickers(self.df):
            key = ('Close', ticker)
            closes = self.df.get(key).reindex(pd.date_range(start, end)).fillna(method='bfill')
            return closes.reindex(pd.date_range(start, end, freq=f))
        else:
            self.df = update_dataframe_ticker(self.df, [ticker])
            key = ('Close', ticker)
            closes = self.df.get(key).reindex(pd.date_range(start, end)).fillna(method='bfill')
            return closes.reindex(pd.date_range(start, end, freq=f))

    def get_closes(self, tickers, start, end, f):
        d = {}
        for ticker in tickers:
            d.update({ticker : self.get_close(ticker, start, end, f)})
        return pd.DataFrame(d)

    # like get_close but it takes a ticker and index instead
    # this code is very sloppy, but I am not used to overloading methods in python and from what I know,
    # trying to make the inputs variable would make the code significantly harder to understand.
    def get_close_index(self, tickers, idx):
        d = {}
        for ticker in tickers:
            key = ('Close', ticker)
            try:
                s = self.df.get(key).reindex(pd.date_range(idx[0], idx[-1])).fillna(method='bfill').reindex(idx)
                d.update({ticker: s})
            except AttributeError:
                print(ticker + ' is unavailable at the dates requested')
        return pd.DataFrame(d)

    def get_percent_change(self, ticker, start, end, f):
        closes = self.get_close(ticker, start, end, f)
        costs = closes.shift(periods=1)
        r = (closes/costs)-1
        return r.fillna(value=0)

    def get_percent_changes(self, tickers, start, end, f):
        closes = self.get_closes(tickers, start, end, f)
        costs = closes.shift(periods=1, fill_value=np.nan)
        return (closes/costs - 1).fillna(value=0)

    # this method is similar to the get_percent_change method, but it takes a ticker and index instead
    def get_percent_change_index(self, tickers, index):
        close = self.get_close_index(tickers, index)
        cost = close.shift(periods=1, fill_value=np.nan)
        return (close / cost - 1).fillna(value=0)

    def get_return(self, ticker, start, end, f):
        closes = self.get_close(ticker, start, end, f)
        cost = closes[0]
        r = closes/cost - 1
        return r

    def get_returns(self, tickers, start, end, f):
        closes = self.get_closes(tickers, start, end, f)
        cost = closes.iloc[0]
        return (closes/cost - 1)

    # this method is similar to the get_return method, but it takes a ticker and index instead
    def get_return_index(self, tickers, index):
        close = self.get_close_index(tickers, index)
        cost = close.iloc[0]
        return close.apply(lambda x: x / cost - 1, axis=1)

    def calc_cov(self, t1, t2, start, end, f):
        df = self.get_percent_changes([t1, t2], start, end, f)
        c = df.cov()
        return c[t1].get(t2)

    def calc_beta_historical(self, ticker, start, end, f):
        mr = self.get_percent_change('^GSPC', start, end, f)
        r = self.get_percent_change(ticker, start, end, f)
        df = pd.DataFrame()
        df[ticker] = r
        df['^GSPC'] = mr
        lm = linear_model.LinearRegression()
        lm.fit(pd.DataFrame(mr), pd.DataFrame(r))
        return lm.coef_[0]

    # this method utilizes the idea that a company's beta is a mean reverting property. What I want it to do
    # is create a dataframe of the company's historical beta's in 3 month intervals for the past 10 years or the entire
    # company's life cycle. Then, I want to return the final value for the moving average. This reflects the mean reverting
    # property -- that a company's beta reverts to it's mean in the long run. However, I do not know enough of the
    # theoretical explanation besides that it relies on the idea of an efficient market in the long run, meaning
    # that it is hinged on EMH
    def calc_beta_adjusted(self, ticker, date):
        idx = pd.date_range(date - datetime.timedelta(days=365 * 10), date, freq='Q')
        list_betas = []
        dates = []
        for i in range(len(idx)):
            if i >= 3:
                start = idx[i - 3]
                end = idx[i]
                beta = self.calc_beta_historical(ticker, start, end, 'BM')
                if beta != [0.0]:
                    list_betas.append(beta)
                    dates.append(end)
        rolling_period = math.floor(1 / 3 * len(dates))
        return pd.Series(list_betas, index=dates).rolling(window=rolling_period).mean()[-1]


    def create_historical_cov_matrix(self, tickers, start, end, f):
        df = pd.DataFrame()
        for ticker in tickers:
            l = self.get_percent_change(ticker, start, end, f)
            df[ticker] = l
        return df.cov()

    def average_return(self, ticker, start, end, f):
        return self.get_return(ticker, start, end, f).mean()

    def average_change(self, ticker, start, end, f):
        return self.get_percent_change(ticker, start, end, f).mean()

    def momentum(self, ticker, start, end, delta):
        close = self.get_close(ticker, start, end, 'B')
        shifted_close = close.shift(periods=delta)
        return close - shifted_close

    def get_trading_volume(self, ticker, start, end, f):
        if ticker in get_avail_tickers(self.df):
            key = ('Volume', ticker)
            return self.df.get(key).reindex(pd.date_range(start, end, freq=f))
        else:
            self.df = update_dataframe_ticker(self.df, [ticker])
            key = ('Close', ticker)
            return self.df.get(key).reindex(pd.date_range(start, end)).fillna(method='bfill')

    # given a list of tickers, returns the available tickers during the specified time period
    def ret_avail_tickers(self, start, end, tickers):
        df = pd.DataFrame()
        for ticker in tickers:
            df[ticker] = self.df.get(('Close', ticker)).loc[start: end]
        return list(df.dropna(inplace=False, axis=1).columns)


def get_closest_date(date, index):
    closest_date = index[0]
    try:
        closest_date = index[0]
        for d in index:
            if abs(date-d) < abs(date-closest_date):
                closest_date = d
        return closest_date
    except TypeError:
        closest_date = index[0].date()
        for d in index:
            d = d.date()
            if abs(date-d) < abs(date-closest_date):
                closest_date = d
        return closest_date

