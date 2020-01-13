import pickle
import pandas as pd
from basic_analysis_toolkit import BasicAnalysis
from company_toolkit import *
import datetime
import numpy as np



# since I've been working heavily with factor models, this class is structured to create a RiskModel from data
# generated from a factor model
# needs:
# - analysis_object: so that the object doesn't depend on an outside object for functionality from BasicAnalysis
class RiskModel:
    def __init__(self, analysis_object, company_dict, market_costs, get_sensitivity, start, end):
        self.analysis = analysis_object
        self.company_dict = company_dict
        # market_costs must be a function in the form of
        # (start, end, analysis, company_dict) -> dataframe of costs
        self.market_costs = market_costs
        # sensitivity must be a function in the form of
        # (date, analysis, company_dict, tickers)
        self.sensitivity = get_sensitivity
        info = self.market_costs(start, end, analysis_object, company_dict)
        self.complete_mc = info.get('mc')
        # complete_sensitivity is a dictionary of the sensitivities used to calculate the market costs.
        # the point of saving this as a field is to cut down on the processing time needed to perform calculations
        # once the RiskModel object is created.
        self.complete_sensitivity = info.get('betas')

        # if the stock is in the sp500, then just return the value for the risk formula

    # this function returns the risk of a portfolio object. Additionally, this is an other way of measuring the risk
    # of an individual company
    # the value that it returns is the variance of the portfolio.
    def portfolio_risk_systematic(self, date, portfolio):
        start = date - datetime.timedelta(days=365)
        end = date
        mc = self.complete_mc[start: end]
        portfolio_history = portfolio.get_historical_performance(self.analysis, end)
        initial_values = portfolio_history.iloc[0].drop(labels='Total Value')
        weights = initial_values / initial_values.sum()
        # determining which sensitivity to use
        closest_date = start
        for d in self.complete_sensitivity.keys():
            if d - end > closest_date - end:
                closest_date = d
        betas = self.complete_sensitivity.get(closest_date).loc[portfolio.overview.columns]
        term1 = weights.T.dot(betas)
        sigma = mc.cov()
        term2 = betas.T.dot(weights)
        return term1.values.dot(sigma.values).dot(term2.values)

    # this method calculates the specific risk of a company
    def specific_risk(self, date, tickers):
        start = date - datetime.timedelta(days=365)
        end = date
        mc = self.complete_mc[start: end]
        ers = []
        for date in mc.index:
            betas = self.complete_sensitivity.get(date.date()).loc[tickers]
            costs = mc.loc[date]
            expected_return = betas.values.dot(costs.values)
            ers.append(expected_return)
        er_dataframe = pd.DataFrame(ers, index=mc.index, columns=tickers)
        ar_dataframe = pd.DataFrame()
        for ticker in tickers:
            ar_dataframe[ticker] = self.analysis.get_percent_change(ticker, start, end, 'BM')
        return pd.DataFrame(np.diag(((ar_dataframe - er_dataframe).var() ** .5).values), columns=tickers)

    def portfolio_risk_specific(self, date, portfolio):
        portfolio_history = portfolio.get_historical_performance(self.analysis, date)
        initial_values = portfolio_history.iloc[0].drop(labels='Total Value')
        weights = initial_values / initial_values.sum()
        S = self.specific_risk(date, weights.index)
        return weights.values.T.dot(S.values).dot(weights)

    def mcr_systematic(self, date, portfolio):
        start = date - datetime.timedelta(days=365)
        end = date
        mc = self.complete_mc[start: end]
        total_systematic = self.portfolio_risk_systematic(date, portfolio)
        portfolio_history = portfolio.get_historical_performance(self.analysis, end)
        initial_values = portfolio_history.iloc[0].drop(labels='Total Value')
        weights = initial_values / initial_values.sum()
        # determining which sensitivity to use
        closest_date = start
        for d in self.complete_sensitivity.keys():
            if d - end > closest_date - end:
                closest_date = d
        betas = self.complete_sensitivity.get(closest_date).loc[portfolio.overview.columns]
        sigma = mc.cov()
        MCR_vector = (betas.values.dot(sigma.values).dot(betas.values.T).dot(weights.values)) / total_systematic
        return pd.Series(MCR_vector, index=weights.index)

    def mcr_specific(self, date, portfolio):
        start = date - datetime.timedelta(days=365)
        end = date
        mc = self.complete_mc[start: end]
        portfolio_history = portfolio.get_historical_performance(self.analysis, end).drop(axis=1, labels='Total Value')
        initial_values = portfolio_history.iloc[0]
        total_specific = self.portfolio_risk_specific(date, portfolio)
        S = self.specific_risk(date, portfolio.overview.columns)
        weights = initial_values / initial_values.sum()
        MCR_vector = (S.values.dot(weights.values)) / total_specific
        return pd.Series(MCR_vector, index=portfolio.overview.columns)

    def mcr_total(self, date, portfolio):
        return self.mcr_systematic(date, portfolio) + self.mcr_specific(date, portfolio)

# look into this code
# figure out the point
    # def group_mcr(self, date, portfolio, t, MCR):
    #     return t.values.T.dot(MCR)

    def contribution_risk(self, date, portfolio, MCR):
        portfolio_history = portfolio.get_historical_performance(self.analysis, date)
        initial_values = portfolio_history.iloc[0].drop(labels='Total Value')
        weights = initial_values / initial_values.sum()
        return weights * MCR

    # theoretically, this value represents the required rate of returns for a company based on the sensitivities
    # to the indexes and the cost of sensitivities
    def expected_return(self, date, tickers):
        s = self.sensitivity(date, self.analysis, self.company_dict, tickers)


