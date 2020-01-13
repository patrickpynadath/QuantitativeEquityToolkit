import pickle
import pandas as pd
from basic_analysis_toolkit import BasicAnalysis
from company_toolkit import *
import datetime
import numpy as np
from sklearn import linear_model
from risk_model_toolkit import RiskModel
risk_free_rates = pd.read_csv('treasury_3month_interest_rate.csv', parse_dates = ['DATE'], index_col = 'DATE').reindex(pd.date_range(datetime.date(day=1, month=1, year=2000), datetime.date(day=1, month=1, year=2020), freq='B'))
sp500_dataframe = pickle.load(open('stock_dataframe.pickle', 'rb'))
analysis = BasicAnalysis(sp500_dataframe)
company_dict = pickle.load(open('company_dictionary.pickle', 'rb'))


def get_size_data(company_dict, tickers):
    sizes = []
    list_ignored_companies = []
    idx = []
    for ticker in tickers:
        try:
            company = company_dict.get(ticker)
            size = company.employees()
            sizes.append(size)
            idx.append(ticker)
        except ValueError:
            list_ignored_companies.append(ticker)
    return pd.Series(sizes, index = idx)

def get_pb_data(company_dict, start, tickers):
    pb_ratios = []
    idx = []
    list_ignored_companies = []
    for ticker in tickers:
        try:
            company = company_dict.get(ticker)
            ratios = company.price_book().sort_index(ascending=False)
            for date in ratios.index:
                if date.year <= start.year: # the information must be available BEFORE the testing period
                    ratio = ratios.get(date)
                    break
                else:
                    ratio = np.nan
            pb_ratios.append(ratio)
            idx.append(ticker)
        except IndexError:
            list_ignored_companies.append(ticker)
    return pd.Series(pb_ratios, index = idx)

# I am omitting earnings_yield because I dont think it is the most relevant to incorporate. The market cost not only
# fluctuates, but also I only have the most recent data, so I can not test this on historical data.
# momentum column
def get_momentum_data(analysis, date, tickers):
    momentums = []
    idx = []
    for ticker in tickers:
        momentum = analysis.momentum(ticker, date - datetime.timedelta(days=60), date, 1).mean()
        idx.append(ticker)
        momentums.append(momentum)
    return pd.Series(momentums, index = idx)

# growth column
def get_growth_data(company_dict, start, tickers):
    growth_list = []
    idx = []
    for ticker in tickers:
        company = company_dict.get(ticker)
        growths = get_growth(company.income['Gross Profit']).sort_index(ascending=False)
        for date in growths.index:
            if date.year <= start.year: # the information must be available BEFORE the testing period
                growth = growths.get(date)
                break
            else:
                growth = np.nan
        growth_list.append(growth)
        idx.append(ticker)
    return pd.Series(growth_list, index=idx)

# earning variability column
def get_earning_var_data(company_dict, tickers):
    # earning variability column
    earning_vars = []
    idx = []
    for ticker in tickers:
        company = company_dict.get(ticker)
        earning_variability = company.earnings_variability()
        earning_vars.append(earning_variability)
        idx.append(ticker)
    return pd.Series(earning_vars, index = idx)

# financial leverage column
def get_fin_leverage_data(company_dict, tickers, start):
    values = []
    idx = []
    for ticker in tickers:
        company = company_dict.get(ticker)
        fin_leverage_ratios = company.debt_equity().sort_index(ascending=False)
        for date in fin_leverage_ratios.index:
            if date.year <= start.year: # the information must be available BEFORE the testing period
                fin_leverage_ratio = fin_leverage_ratios.get(date)
                break
            else:
                financial_leverage_ratio = np.nan
        values.append(fin_leverage_ratio)
        idx.append(ticker)
    return pd.Series(values, index=idx)

# volatility
def get_vol_data(analysis, tickers, date):
    values = []
    idx = []
    for ticker in tickers:
        volatility = (analysis.get_close(ticker, date-datetime.timedelta(days=60), date, 'B').var() ** .5)/analysis.get_close(ticker, date-datetime.timedelta(days=60), date, 'B').mean()
        idx.append(ticker)
        values.append(volatility)
    return pd.Series(values, index=idx)

# trading activity
def get_trading_activity(analysis, tickers, date):
    values = []
    idx = []
    for ticker in tickers:
        volume = analysis.get_trading_volume(ticker, date-datetime.timedelta(days=60), date, 'BM').mean()
        values.append(volume)
        idx.append(ticker)
    return pd.Series(values, index=idx)

# this function returns a dataframe of the cross sectional sensitivities to the indices.
# it takes one date, which represents the cutoff date for information. In other words, the sensitivities
# are based on data until the date inputted. This ensures I am not 'overtraining' the risk model by having it
# look at information that would during the testing period of the model
def create_frm_sensitivities(date, analysis, company_dict, tickers):
    sensitivities = pd.DataFrame()
    sensitivities['Size'] = get_size_data(company_dict, tickers)
    sensitivities['Price to Book'] = get_pb_data(company_dict, date, sensitivities.index)
    sensitivities['Momentum'] = get_momentum_data(analysis, date, sensitivities.index)
    sensitivities['Growth'] = get_growth_data(company_dict, date, sensitivities.index)
    sensitivities['Earning Variability'] = get_earning_var_data(company_dict, sensitivities.index)
    sensitivities['Financial Leverage'] = get_fin_leverage_data(company_dict, sensitivities.index, date)
    sensitivities['Volatility'] = get_vol_data(analysis, sensitivities.index, date)
    sensitivities['Trading Activity'] = get_trading_activity(analysis, sensitivities.index, date)
    df = sensitivities.replace([np.inf, -np.inf], np.nan)
    return (df - df.mean()) / (df.max() - df.min())

def get_return_data(tickers, analysis, start, end):
    return_data = pd.DataFrame()
    for ticker in tickers:
        data = analysis.get_percent_change(ticker, start, end, 'BM')
        return_data[ticker] = data
    return return_data

# this function creates a dataframe of the market costs (premia) for each time period.
# update: this function will also return the dictionary of sensitivty dataframes so that
# they do not need to be recreated as that wastes time.
def create_frm_market_costs(start, end, analysis, company_dict):
    # create empty list for the coefs to be placed into
    costs = []
    for i in range(10):
        costs.append([])
    daterange = pd.date_range(start, end, freq='BM') # the frequency is set to BM intentionally
    return_data = get_return_data(company_dict.keys(), analysis, start, end).transpose()
    sensitivity_dict = {}
    for date in daterange:
        frm_sensitivities = create_frm_sensitivities(date, analysis, company_dict, company_dict.keys())
        frm_sensitivities.dropna(inplace=True)
        sensitivity_dict.update({date.date() : frm_sensitivities})
        r = return_data.reindex(frm_sensitivities.index)
        # rf rate calculation
        rf = risk_free_rates['TB3MS'].get(date)
        if str(rf) == 'nan' or str(rf) == 'None':
            rf = .03 # manually entering this, given in the actual data set mapped to 8/01/2014
        rf = (1+rf) ** (1/12) - 1
        X = frm_sensitivities
        Y = r[date] - rf
        regr = linear_model.LinearRegression(fit_intercept=False)
        regr.fit(X, Y)
        for i in range(len(regr.coef_)):
            costs[i].append(regr.coef_[i])
        print(date)
    list_series = []
    for list_costs in costs:
        if len(list_costs) != 0:
            s = pd.Series(list_costs, index=return_data.columns)
            list_series.append(s)
    market_costs_indices = pd.DataFrame()
    market_costs_indices['Cost of Size'] = list_series[0]
    market_costs_indices['Cost of PB Ratio'] = list_series[1]
    market_costs_indices['Cost of Momentum'] = list_series[2]
    market_costs_indices['Cost of Growth'] = list_series[3]
    market_costs_indices['Cost of Earning Variability'] = list_series[4]
    market_costs_indices['Cost of Financial Leverage'] = list_series[5]
    market_costs_indices['Cost of Volatility'] = list_series[6]
    market_costs_indices['Cost of Trading Volume'] = list_series[7]
    return {'mc' : market_costs_indices, 'betas' : sensitivity_dict}


def create_frm(analysis, company_dict, start, end):
    market_costs = create_frm_market_costs
    sensitivities = create_frm_sensitivities
    return RiskModel(analysis, company_dict, market_costs, sensitivities, start, end)


def create_frm_sensitivities_unnormalized(date, analysis, company_dict, tickers):
    sensitivities = pd.DataFrame()
    sensitivities['Size'] = get_size_data(company_dict, tickers)
    sensitivities['Price to Book'] = get_pb_data(company_dict, date, sensitivities.index)
    sensitivities['Momentum'] = get_momentum_data(analysis, date, sensitivities.index)
    sensitivities['Growth'] = get_growth_data(company_dict, date, sensitivities.index)
    sensitivities['Earning Variability'] = get_earning_var_data(company_dict, sensitivities.index)
    sensitivities['Financial Leverage'] = get_fin_leverage_data(company_dict, sensitivities.index, date)
    sensitivities['Volatility'] = get_vol_data(analysis, sensitivities.index, date)
    sensitivities['Trading Activity'] = get_trading_activity(analysis, sensitivities.index, date)
    return sensitivities.replace([np.inf, -np.inf], np.nan)