import requests
import pandas as pd
import lxml
from lxml import html
import DataRetrieval
import pickle
from bs4 import *
import datetime
import math
import numpy as np

risk_free_rates = pd.read_csv('treasury_3month_interest_rate.csv', parse_dates = ['DATE'], index_col = 'DATE').reindex(pd.date_range(datetime.date(day=1, month=1, year=2000), datetime.date(day=1, month=1, year=2020), freq='B'))

def get_profile_info(t):
    url = 'https://finance.yahoo.com/quote/' + t + '/profile?p=' + t
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    parsed_info = []
    for tag in soup.find_all('p'):
        if tag['class'][0] == 'D(ib)':
            if tag['class'][1] == 'Va(t)':
                for child in tag.children:
                    if child.name == 'span':
                        parsed_info.append(child.text)
    sector = parsed_info[1]
    industry = parsed_info[3]
    employees = parsed_info[5]
    return {'Sector': sector, 'Industry': industry, 'Employees': employees}



def get_balance(ticker):
    url = 'https://finance.yahoo.com/quote/' + ticker + '/balance-sheet?p=' + ticker

    # Fetch the page that we're going to parse
    page = requests.get(url)

    # Parse the page with LXML, so that we can start doing some XPATH queries
    # to extract the data that we want
    tree = html.fromstring(page.content)

    # Using XPATH, fetch all table elements on the page
    table = tree.xpath('//table')

    # Ensure that there is only one table on the page; if there is more than
    # one table, then it's possible that Yahoo Finance has changed their page
    # structure. If this fails, please let me know which symbol you're trying
    # to scrape in the comments below
    assert len(table) == 1

    # Now that we've got the table element, convert it back to a string,
    # so that it can be parsed by Pandas
    tstring = lxml.etree.tostring(table[0], method='html')

    # Read the HTML table into a Pandas DataFrame - read_html
    # is designed to read HTML tables into a list of dataframe objects,
    # one dataframe for each table.
    df = pd.read_html(tstring)[0]
    df = df.transpose()
    df.columns = df.iloc[0]
    df = df[1:]
    df = df.set_index(pd.to_datetime(df['Period Ending'],format='%m/%d/%Y'))
    df = df.drop(columns=['Period Ending', 'Current Assets'])
    return df


def get_income(ticker):
    url = 'https://finance.yahoo.com/quote/' + ticker + '/financials?p=' + ticker
    page = requests.get(url)
    tree = html.fromstring(page.content)
    table = tree.xpath('//table')
    assert len(table) == 1
    tstring = lxml.etree.tostring(table[0], method='html')
    df = pd.read_html(tstring)[0]
    df = df.transpose()
    df.columns = df.iloc[0]
    df = df[1:]
    df = df.set_index(pd.to_datetime(df['Revenue'],format='%m/%d/%Y'))
    df = df.drop(columns=['Revenue', 'Non-recurring Events'])
    return df


def get_cashflow(ticker):
    url = 'https://finance.yahoo.com/quote/' + ticker + '/cash-flow?p=' + ticker
    page = requests.get(url)
    tree = html.fromstring(page.content)
    table = tree.xpath('//table')
    assert len(table) == 1
    tstring = lxml.etree.tostring(table[0], method='html')
    df = pd.read_html(tstring)[0]
    df = df.transpose()
    df.columns = df.iloc[0]
    df = df[1:]
    df = df.set_index(pd.to_datetime(df['Period Ending'],format='%m/%d/%Y'))
    df = df.drop(columns=['Period Ending', 'Operating Activities, Cash Flows Provided By or Used In', 'Investing Activities, Cash Flows Provided By or Used In', 'Financing Activities, Cash Flows Provided By or Used In'])
    return df


def get_stats_table(ticker):
    url = 'https://finance.yahoo.com/quote/' + ticker + '/key_statistics'
    page = requests.get(url)
    tree = html.fromstring(page.content)
    table = tree.xpath('//table')
    tstring = lxml.etree.tostring(table[1], method='html')
    df = pd.read_html(tstring)[0]
    df = df.transpose()
    df.columns = df.iloc[0]
    df = df[1:]
    return df


def dictionary_companies():
    sp500_tickers = DataRetrieval.get_sp500_tickers(DataRetrieval.get_sp500_table())
    d = {}
    removed_companies = []
    for ticker in sp500_tickers:
        print(ticker)
        try:
            comp = Company(ticker)
            pair = {ticker: comp}
            d.update(pair)
        except IndexError:
            print("Error with Internet Speed, removing company and adding it to removed company list")
            removed_companies.append(ticker)
    pickle.dump(removed_companies, open('removed_companies.pickle', 'wb'))
    return d


def get_company_dict():
    d = pickle.load(open('company_dictionary.pickle', 'rb'))
    return d


def dump_company_dict(d):
    pickle.dump(d, open('company_dictionary.pickle', 'wb'))


class Company:
    def __init__(self, ticker):
        self.ticker = ticker
        self.cash_flow = get_cashflow(ticker)
        self.balance = get_balance(ticker)
        self.income = get_income(ticker)
        self.stats_table = get_stats_table(ticker)
        self.info = get_profile_info(ticker)

    def market_cap(self):
        price = self.stats_table['Market Cap'].get(1)
        if 'B' in price:
            price = price.replace('B', '')
            price = float(price) * 1000000000
        elif 'M' in price:
            price = price.replace('M', '')
            price = float(price) * 1000000
        elif 'T' in price:
            price = price.replace('T', '')
            price = float(price) * 1000000000000
        return price

    def price_book(self):
        #total_assets = float(self.balance['Total Assets'][-1]) * 1000  # total assets
        total_assets = format_column(self.balance['Total Assets'])
        price = self.market_cap()
        return price / total_assets

    def employees(self):
        return float(self.info.get('Employees').replace(',', ''))

    def price_earnings(self):
        return float(self.stats_table['PE Ratio (TTM)'].get(1))

    def forward_yield(self):
        value = self.stats_table['Forward Dividend & Yield'].get(1).split()[1].replace('(', '').replace(')', '').replace('%', '')
        if 'N/A' in value:
            return 0
        else:
            return float(value)

    # I am measuring the earnings variability of a company by finding the standard deviation of the earnings applicable to
    # common shares. I then return the ratio of the standard deviation to the mean in order to capture the variability
    def earnings_variability(self):
        earnings_shares = pd.to_numeric(self.income['Net Income Applicable To Common Shares'], errors='coerce').fillna(value=np.nan)
        return (earnings_shares.var() ** .5)/(earnings_shares.mean())

    def debt_equity(self):
        current_liabilities = format_column(self.balance['Total Current Liabilities'])
        long_term_debt = format_column(self.balance['Long Term Debt'])
        liabilities = (current_liabilities + long_term_debt)
        shareholders_equity = format_column(self.balance['Total stockholders\' equity'])
        return liabilities / shareholders_equity

    def equity_multiplier(self):
        total_assets = format_column(self.balance['Total Assets'])
        total_equity = format_column(self.balance['Total stockholders\' equity'])
        return total_assets / total_equity

    def degree_fin_lev(self):
        earnings_shares = format_column(self.income['Net Income Applicable To Common Shares'])
        earnings_change = (earnings_shares - earnings_shares.shift(periods=1)) / earnings_shares.shift(periods=1)
        ebit = format_column(self.income['Earnings Before Interest and Taxes'])
        ebit_change = (ebit - ebit.shift(periods=1)) / ebit.shift(periods=1)
        return earnings_change / ebit_change

    # getting the sector and industry columns
    # make abstract function for making a series with all the companies in the list from info from company_dict
    # f must be some function taht takes a ticker and returns some value
    # determining the required rate of return
    # CAPM method
    # issue: it is very time sensitive. I want it to be resistent to short term changes.
    # idea: possibly take the average monthly return, raise it to the 12th power to get the expected return
    # this makes sense because I am not trying to capture the immediate required rate of return, I am trying to capture
    # the expected required rate of return. I believe that using the estimator defined by this procedure will be
    # a better estimator than taking the overall returns because that can change very easily due to temporary
    # systematic shocks. This insulates this value from those types of shocks. I want to this value to insulated from
    # those types of risks because the RRR is a value used in valuation, which places emphasis on the intrinsic worth.
    # having a value that is highly correlated with external factors means that it is not intrinsic, but rather extrensic
    # this is for the cohesiveness of the company_toolkit, which seeks to develop tools for valuation methods.

    def calc_rrr_capm(self, date, analysis):
        # for calculating expected market return
        start = date - datetime.timedelta(days=365 * 5)
        end = date
        mr = (1 + analysis.get_percent_change('^GSPC', start, end, 'BM').mean()) ** 12 - 1
        beta = analysis.calc_beta_adjusted(self.ticker, end)
        rf = risk_free_rates['TB3MS'].get(date)
        if str(rf) == 'nan' or str(rf) == 'None':
            rf = 3  # manually entering this, given in the actual data set mapped to 8/01/2014
        rf = rf * .01
        return rf + beta * (mr - rf)

    # this method calculates the required return given a risk model. However, I think that the flaws in my current
    # frm prevent this method from delivering actual results. As a result, I will hold off on using this
    def calc_rrr_rm(self, date, analysis, risk_model):
        rf = risk_free_rates['TB3MS'].get(date)
        if str(rf) == 'nan' or str(rf) == 'None':
            rf = .03  # manually entering this, given in the actual data set mapped to 8/01/2014
        rf = rf * .01
        rolling_period = math.floor(1 / 3 * len(risk_model.complete_mc.index))
        costs = risk_model.complete_mc.rolling(window=rolling_period).mean().iloc[-1]
        closest_date = get_closest_date(date, list(risk_model.complete_sensitivity.keys()))
        betas = risk_model.complete_sensitivity.get(closest_date).loc[self.ticker]
        return rf + ((betas.values.dot(costs.values) + 1) ** 12 - 1)

    # this method will calculate the cost of debt for a company based on the information in their
    # balance statement
    def get_cost_debt(self):
        interest_expense = pd.to_numeric(self.income['Interest Expense'], errors='coerce').fillna(value=0)
        debt = pd.to_numeric(self.balance['Long Term Debt'], errors='coerce').fillna(value=np.nan)
        return -1 * interest_expense / debt

    def get_tax_rate(self):
        ebit = pd.to_numeric(self.income['Earnings Before Interest and Taxes'], errors='coerce')
        tax_expense = pd.to_numeric(self.income['Income Tax Expense'], errors='coerce')
        return tax_expense / ebit

    # calculates the weighted average cost of capital. the formula is from the quantitative equity portfolio management
    # this uses only equity and debt for simplicity
    def calc_wacc(self, date, analysis):
        try:
            closest_date = get_closest_date(date, self.get_cost_debt().index)
            cost_debt = self.get_cost_debt().fillna(value=0)[closest_date]
            tax_rate = self.get_tax_rate().fillna(value=0)[closest_date]
            cost_equity = self.calc_rrr_capm(date, analysis)
            equity = self.market_cap()
            debt = pd.to_numeric(self.balance['Long Term Debt'], errors='coerce').fillna(value=0)[closest_date]
            total_market_value = equity + debt
            return (cost_equity * equity + cost_debt * (1 - tax_rate) * debt) / total_market_value
        except IndexError: # this exception imp;ies that the company doesn't have the data to calculate it's wacc
            # durnig the specific time period
            return np.nan

    def get_wc(self):
        current_assets = pd.to_numeric(self.balance['Total Current Assets'])
        current_liabilities = pd.to_numeric(self.balance['Total Current Liabilities'])
        working_capital = current_assets - current_liabilities
        return working_capital

    def get_ebitda_margin(self):
        rev = pd.to_numeric(self.income['Total Revenue'], errors='coerce')
        operating_exp = pd.to_numeric(self.income['Total Operating Expenses'], errors='coerce')
        ebitda = rev - operating_exp
        return ebitda / rev

    def depreciation_margin(self):
        rev = format_column(self.income['Total Revenue'])
        depr = format_column(self.cash_flow['Depreciation'])
        return depr / rev

    def capex_deltamargin(self):
        revs = pd.to_numeric(self.income['Total Revenue'], errors='coerce').sort_index()
        delta_revs = revs - revs.shift(fill_value=np.nan)
        capex = -1 * pd.to_numeric(self.cash_flow['Capital Expenditure'], errors='coerce').sort_index()
        return capex / delta_revs


    def depr_deltamargin(self):
        revs = pd.to_numeric(self.income['Total Revenue'], errors='coerce').sort_index()
        delta_revs = revs - revs.shift(fill_value = np.nan)
        depr = pd.to_numeric(self.cash_flow['Depreciation'], errors='coerce').sort_index()
        return depr / delta_revs


    def get_delta_wc(self):
        w1 = self.get_wc().sort_index()
        w2 = w1.shift(fill_value=np.nan)
        return w1 - w2

    def get_profitability(self):
        rev = pd.to_numeric(self.income['Total Revenue'], errors='coerce')

        operating_exp = pd.to_numeric(self.income['Total Operating Expenses'], errors='coerce')
        ebitda = rev - operating_exp
        depreciation = pd.to_numeric(self.cash_flow['Depreciation'], errors='coerce')
        other_income = pd.to_numeric(self.income['Total Other Income/Expenses Net'], errors='coerce')
        ebit = ebitda - depreciation + other_income
        delta_revs = rev - rev.shift(fill_value = np.nan)
        tax_rate = self.get_tax_rate().mean()
        nopat = ((1 - tax_rate) * ebit).sort_index()
        delta_nopat = nopat - nopat.shift(fill_value=np.nan)
        # instead of using sales, I will be using revenues overall

        return nopat / rev

    def get_delta_profitability(self):
        rev = pd.to_numeric(self.income['Total Revenue'], errors='coerce')

        operating_exp = pd.to_numeric(self.income['Total Operating Expenses'], errors='coerce')
        ebitda = rev - operating_exp
        depreciation = pd.to_numeric(self.cash_flow['Depreciation'], errors='coerce')
        other_income = pd.to_numeric(self.income['Total Other Income/Expenses Net'], errors='coerce')
        ebit = ebitda - depreciation + other_income
        delta_revs = rev - rev.shift(fill_value=np.nan)
        tax_rate = self.get_tax_rate().mean()
        nopat = ((1 - tax_rate) * ebit).sort_index()
        delta_nopat = nopat - nopat.shift(fill_value=np.nan)
        # instead of using sales, I will be using revenues overall
        return delta_revs/delta_nopat

    def get_operating_assets(self):
        total_assets = pd.to_numeric(self.balance['Total Assets'], errors='coerce').fillna(value=0)
        cash_equiv = pd.to_numeric(self.balance['Cash And Cash Equivalents'], errors='coerce').fillna(value=0)
        short_term_investments = pd.to_numeric(self.balance['Short Term Investments'], errors='coerce').fillna(value=0)
        liabilities = pd.to_numeric(self.balance['Total Liabilities'], errors='coerce').fillna(value=0)
        debt = pd.to_numeric(self.balance['Long Term Debt'], errors='coercce').fillna(value=0)
        short_term_debt = pd.to_numeric(self.balance['Short/Current Long Term Debt'], errors='coerce').fillna(value=0)
        financial_liabilities = liabilities + debt + short_term_debt
        return total_assets - cash_equiv - short_term_investments + financial_liabilities

    def get_operating_liabilities(self):
        delta_wc = self.get_delta_wc()
        capx = pd.to_numeric(self.cash_flow['Capital Expenditure'], errors='coerce').fillna(value=0)
        depreciation = pd.to_numeric(self.cash_flow['Depreciation'], errors='coerce').fillna(value=0)
        return delta_wc + (capx - depreciation)

    def get_scalability(self):
        rev = pd.to_numeric(self.income['Total Revenue'], errors='coerce')
        delta_rev = rev.sort_index() - rev.sort_index().shift(fill_value=np.nan)
        noa = self.get_operating_assets()
        delta_noa = noa.sort_index() - noa.sort_index().shift(fill_value=np.nan)
        return delta_rev / delta_noa


    # this forecasts firm value based on the methedology used to forecast FCFF in chapter 6 of the
    # quantitative equity book
    # the point of this isn't to derive a fair value, but to derive the sensitivities to the fair value.
    # I am still working out a few bugs -- for example, Amazon's sensitivity to profitability is negative,
    # which doesn't intuitively doesn't make sense.
    def forecast_naive_ov(self, date, analysis):
        fcffs = []
        profitability = self.get_profitability().mean()
        scalability = self.get_scalability().mean()
        wacc = self.calc_wacc(date, analysis)
        initial_sales = pd.to_numeric(self.income['Total Revenue'], errors='coerce')[-1]
        growth = get_growth(self.income['Total Revenue']).mean()
        ov = (initial_sales * (profitability - (growth / scalability)) * (1 + growth) / (wacc - growth)) * 1000
        profitability_sens = 1 / (profitability - (growth / scalability))
        sales_sens = 1 / initial_sales
        growth_sens = (-1 / (scalability * (profitability - growth / scalability))) + (1 / (1 + growth)) + (
                    1 / (wacc - growth))
        wacc_sens = (-1 / wacc - growth)
        scalability_sens = (growth / ((scalability ** 2) * (profitability - growth / scalability)))
        return pd.DataFrame({'Operating Value': ov
                , 'wacc': wacc
                , 'growth': growth
                , 'scalability': scalability
                , 'profitability': profitability
                , 'Profitability Sensitivity': profitability_sens
                , 'Sales Sensitivity': sales_sens
                , 'Growth Sensitivity': growth_sens
                , 'Wacc Sensitivity': wacc_sens
                , 'Scalability Sensitivity': scalability_sens}, index=[self.ticker])

    # the point of this method is to provide the same level of information as seen in page 177 of QEPM
    # I want to utilize the breadth of the database when providing this analysis, so the goal is to
    # compare the business not only with it's own historical performance, but the average yearly performance
    # for that businesses sectors.
    # I will be splitting them up into several methods, and recombining them at a later point

    def get_fcffs_historical(self):
        rev = pd.to_numeric(self.income['Total Revenue'], errors='coerce')
        operating_exp = pd.to_numeric(self.income['Total Operating Expenses'], errors='coerce')
        ebitda = rev - operating_exp
        depreciation = pd.to_numeric(self.cash_flow['Depreciation'], errors='coerce')
        other_income = pd.to_numeric(self.income['Total Other Income/Expenses Net'], errors='coerce')
        ebit = ebitda - depreciation + other_income
        tax_rate = self.get_tax_rate().mean()
        nopat = (1 - tax_rate) * ebit
        delta_working_capital = self.get_delta_wc()
        capx = pd.to_numeric(self.cash_flow['Capital Expenditure'], errors='coerce')
        other_investments = pd.to_numeric(self.cash_flow['Other Cash flows from Investing Activities'],
                                          errors='coerce').fillna(value=0)
        fcff = nopat - delta_working_capital - capx - other_investments + depreciation
        df = pd.DataFrame({'Revenue': rev
                              , 'Operating Expense': operating_exp
                              , 'EBITDA': ebitda
                              , 'Depreciation': depreciation
                              , 'Other Income': other_income
                              , 'EBIT': ebit
                              , 'NOPAT': nopat
                              , 'Change in NWC': delta_working_capital
                              , 'CAPX': capx
                              , 'Other Investments': other_investments
                              , 'FCFF': fcff})
        return df

    def sales_analysis(self, company_dict):
        sector_peers = get_sector_peers(company_dict, self.info.get('Sector'))
        comp_rev_growth = get_growth(self.income['Total Revenue'])[-1]
        comp_rev_growth_mean = get_growth(self.income['Total Revenue']).mean()
        sector_avg = get_data(sector_peers,
                              lambda ticker: get_growth(company_dict.get(ticker).income['Total Revenue']).mean()).mean()
        idx = ['Revenue Growth']
        return pd.DataFrame({self.ticker: comp_rev_growth
                                , self.ticker + ' Historical Avg': comp_rev_growth_mean
                                , self.info.get('Sector') + ' Avg': sector_avg}, index=idx)

    def profitability_analysis(self, company_dict):
        # ebitda margin analysis
        sector_peers = get_sector_peers(company_dict, self.info.get('Sector'))

        ebitda_margin = self.get_ebitda_margin()[-1]
        ebitda_margin_avg = self.get_ebitda_margin().mean()
        sector_ebitda_margin_avg = get_data(sector_peers,
                                            lambda ticker: company_dict.get(ticker).get_ebitda_margin().mean()).mean()

        # depr margin analysis
        depr_margin = self.depreciation_margin()[-1]
        depr_margin_avg = self.depreciation_margin().mean()
        sector_depr_margin_avg = get_data(sector_peers,
                                          lambda ticker: company_dict.get(ticker).depreciation_margin().mean()).mean()

        # operating margin analysis
        operating_margin = ebitda_margin - depr_margin
        operating_margin_avg = ebitda_margin_avg - depr_margin_avg
        sector_operating_margin_avg = sector_ebitda_margin_avg - sector_depr_margin_avg

        # tax rate analysis
        tax_rate = self.get_tax_rate()[-1]
        tax_rate_avg = self.get_tax_rate().mean()
        sector_tax_rate_avg = get_data(sector_peers, lambda ticker: company_dict.get(ticker).get_tax_rate().mean()).mean()

        # nopat margin
        nopat_margin = (1 - tax_rate) * operating_margin
        nopat_margin_avg = (1 - tax_rate_avg) * operating_margin_avg
        sector_nopat_margin_avg = (1 - sector_tax_rate_avg) * sector_operating_margin_avg

        idx = ['EBITDA Margin', 'Depr/Sales', 'Operating Margin', 'Tax Rate', 'NOPAT Margin']
        return pd.DataFrame({self.ticker + ' Estimate': [ebitda_margin, depr_margin, operating_margin, tax_rate
            , nopat_margin]
                                ,
                             self.ticker + ' 5 yr hist avg': [ebitda_margin_avg, depr_margin_avg, operating_margin_avg,
                                                              tax_rate_avg
                                 , nopat_margin_avg]
                                , self.info.get('Sector') + ' Avg': [sector_ebitda_margin_avg, sector_depr_margin_avg,
                                                                     sector_operating_margin_avg,
                                                                     sector_tax_rate_avg, sector_nopat_margin_avg]
                             }, index=idx)

    def scalability_analysis(self, company_dict):
        sector_peers = get_sector_peers(company_dict, self.info.get('Sector'))
        # capex_margin analysis
        capex_margin = self.capex_deltamargin()[-1]
        capex_margin_avg = self.capex_deltamargin().mean()
        sector_capex_margin_avg = get_data(sector_peers,
                                           lambda ticker: company_dict.get(ticker).capex_deltamargin().mean()).mean()

        # depr/delta_sales analysis
        depr_margin = self.depr_deltamargin()[-1]
        depr_margin_avg = self.depr_deltamargin().mean()
        sector_depr_margin = get_data(sector_peers,
                                      lambda ticker: company_dict.get(ticker).depr_deltamargin().mean()).mean()

        # icapex analysis
        icapex_margin = capex_margin - depr_margin
        icapex_margin_avg = capex_margin_avg - depr_margin_avg
        sector_icapex_margin_avg = sector_capex_margin_avg - sector_depr_margin

        # scalability analysis
        scalability = self.get_scalability()[-1]
        scalability_avg = self.get_scalability().mean()
        sector_scalability_avg = get_data(sector_peers, lambda ticker: company_dict.get(ticker).get_scalability().mean()).mean()

        idx = ['Capex/delta sales', 'Depr/delta sales', 'ICAPEX/delta sales', 'Scalability']
        return pd.DataFrame({self.ticker + ' Estimate': [capex_margin, depr_margin, icapex_margin, scalability]
                            ,self.ticker + ' Hist Avg': [capex_margin_avg, depr_margin_avg, icapex_margin_avg, scalability_avg]
                                , self.info.get('Sector') + ' Avg': [sector_capex_margin_avg, sector_depr_margin,
                                                                     sector_icapex_margin_avg,
                                                                     sector_scalability_avg]},
                            index=idx)

    # the issue that I am running into has to deal with the WACC calculations. Some companies throw an error
    # because they are not available during that time. I think the best solution to this problem is to
    # create
    def eva_analysis(self, date, analysis, company_dict):
        idx = [self.ticker + ' Estimate', self.ticker + ' Historical Avg', self.info.get('Sector') + ' Avg']
        sector_peers = get_sector_peers(company_dict, self.info.get('Sector'))
        profitability = self.profitability_analysis(company_dict)
        prof = profitability.loc['NOPAT Margin']
        scalability = self.scalability_analysis(company_dict)
        scal = scalability.loc['Scalability']
        ric = scal.values * prof.values

        # wacc analysis
        wacc_est = self.calc_wacc(date, analysis)
        wacc_avg = np.nan
        wacc_sector = get_data(sector_peers, lambda ticker: company_dict.get(ticker).calc_wacc(date, analysis)).mean()

        idx = ['RIC', 'WACC']
        df = pd.DataFrame({self.ticker + ' Estimate': [ric[0], wacc_est]
                              , self.ticker + ' Historical Avg': [ric[1], wacc_avg]
                              , company_dict.get(self.ticker).info.get('Sector') + ' Avg': [ric[2], wacc_sector]},
                          index=idx)
        return df

    # the growth should be in the form of a series indexed by the dates.
    # the dates should

    def get_nopat(self):
        rev = pd.to_numeric(self.income['Total Revenue'], errors='coerce')
        operating_exp = pd.to_numeric(self.income['Total Operating Expenses'], errors='coerce')
        ebitda = rev - operating_exp
        depreciation = pd.to_numeric(self.cash_flow['Depreciation'], errors='coerce')
        other_income = pd.to_numeric(self.income['Total Other Income/Expenses Net'], errors='coerce')
        ebit = ebitda - depreciation + other_income
        tax_rate = self.get_tax_rate().mean()
        return (1 - tax_rate) * ebit

    def get_deltanoa(self):
        noa = self.get_operating_assets()
        return noa.sort_index() - noa.sort_index().shift(fill_value=np.nan)


def format_column(column):
    return 1000 * pd.to_numeric(column, errors='coerce').fillna(value=0).sort_index()


def get_growth(column):
    values = format_column(column)
    change = (values - values.shift(periods=1))/(values.shift(periods=1))
    return change

# this function, given a list of tickers, returns the information that is obtained when fucntion f is performed
# on the ticker
def get_data(tickers, f):
    values = []
    for ticker in tickers:
        values.append(f(ticker))
    return pd.Series(values, index=tickers)


# this method is to make it easier to graph the information in the 3 statements within a company object
# since the scraping method stores the info as strings.
def graph_data(df):
    new_df = pd.DataFrame()
    for column in df.columns:
        new_df[column] = pd.to_numeric(df[column], errors='coerce')
    return new_df


# the difference between the above function and this one is that this takes an f that returns a Series for that company
# this returns a DataFrame of all the series.
def compare_values(tickers, f):
    d = {}
    for ticker in tickers:
        s = f(ticker)
        d.update({ticker : s})
    return pd.DataFrame(d)


def get_closest_date(date, index):
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


#this function returns the list of company's in a given sector
def get_sector_peers(company_dict, sector):
    sector_peers = []
    for ticker in list(company_dict.keys()):
        if company_dict.get(ticker).info.get('Sector') == sector:
            sector_peers.append(ticker)
    return sector_peers


def forecast_lineitem(lineitem, dates):
    lineitem = pd.to_numeric(lineitem, errors='coerce')
    growth = get_growth(lineitem).mean()
    new_items = []
    initial_item = lineitem.sort_index()[-1]
    for date in dates:
        forecasted_item = (1 + growth) * initial_item
        new_items.append(forecasted_item)
        initial_item = forecasted_item
    return pd.Series(new_items, index=dates)


def forecast_dates(start_date):
    dates = []
    for i in range(1, 6):
        dates.append(start_date + datetime.timedelta(days=365 * i))
    return dates


def calc_discount_rate(wacc, dates):
    discounts = []
    for i in range(1, len(dates) + 1):
        discounts.append(1 / (1 + wacc) ** i)
    return pd.Series(discounts, index=dates)


# this function creates a series of values indexed by the given date
def create_constant_series(value, dates):
    items = []
    for date in dates:
        items.append(value)
    return pd.Series(items, index=dates)


class RIC_DCF:
    def __init__(self, ticker, company_dict, analysis, date, fade_rate, lgr):
        self.company = company_dict.get(ticker)
        self.wacc = self.company.calc_wacc(date, analysis)

        self.explicit_period = self.dcf_explicit_period(date, analysis)
        self.fade_period = self.dcf_fade_period(date, analysis, fade_rate, lgr)

        start_date = self.explicit_period.index[0]
        end_date = self.fade_period.index[-1]
        end_fcff = self.fade_period['FCFFs (Real)'][-1]
        self.terminal_value = self.calc_terminal_value(end_fcff, start_date, end_date, lgr, self.wacc)

    def dcf_explicit_period(self, date, analysis):
        new_dates = forecast_dates(self.company.income.sort_index().index[-1])
        sales_growth = get_growth(self.company.income['Total Revenue']).mean()
        profitability = self.company.get_profitability().mean()
        scalability = self.company.get_scalability().mean()
        estimated_RIC = profitability * scalability
        nopat = self.company.get_nopat()
        delta_noa = self.company.get_deltanoa()
        growth = create_constant_series(sales_growth, new_dates)
        ric = create_constant_series(estimated_RIC, new_dates)
        # forecasting delta_noa and nopat
        initial_nopat = nopat.sort_index()[-1]
        initial_deltanoa = delta_noa[-1]
        new_nopats = []
        new_deltanoas = []
        for date in new_dates:
            new_nopat = initial_nopat + initial_deltanoa * estimated_RIC
            # Here I run into a bit of an issue: this overestimates the actual change in net operating assets
            # by almost a factor of two. While this is the "economically" correct way to calculate it
            # based on the company's growth, I really don't think that this leads to accurate values. My solution is to
            # assume that they use the average of the past 4 years throughout the explicit period. I have commented
            # out the actual formulat that is technically correct
            #new_deltanoa = initial_nopat * growth.get(date) / ric.get(date)
            new_deltanoa = delta_noa.mean()
            new_nopats.append(new_nopat)
            new_deltanoas.append(new_deltanoa)
            initial_nopat = new_nopat
            initial_deltanoa = new_deltanoa
        new_nopats = pd.Series(new_nopats, index=new_dates)
        new_deltanoas = pd.Series(new_deltanoas, index=new_dates)
        new_fcffs = new_nopats - new_deltanoas
        discount_rates = calc_discount_rate(self.wacc, new_dates)
        present_values = new_fcffs * discount_rates
        return pd.DataFrame({'Sales Growth': growth
                              , 'RIC': ric
                              , 'NOPAT': new_nopats
                              , 'Delta NOA': new_deltanoas
                              , 'FCFFs (Nominal)': new_fcffs
                              , 'Discount Rates': discount_rates
                              , 'FCFFs (Real)': present_values})


    def dcf_fade_period(self, date, analysis, fade_rate, lgr):
        initial_ric = self.explicit_period['RIC'].mean()
        initial_sales_growth = self.explicit_period['Sales Growth'].mean()
        initial_date = self.explicit_period.index[-1]
        initial_nopat = self.explicit_period['NOPAT'][-1]
        initial_deltanoa = self.explicit_period['Delta NOA'][-1]
        wacc = self.company.calc_wacc(date, analysis)
        initial_difference_ric = wacc - initial_ric
        initial_difference_growth = initial_sales_growth - lgr
        if initial_difference_growth > 0:
            direction_g = -1
        else:
            direction_g = 1
        # creating the series for the RIC fade
        i = 0
        ric_list = []
        g_list = []
        date_list = []
        while round(initial_ric, 2) != round(wacc, 2):
            increment_ric = fade_rate * initial_difference_ric
            initial_ric = initial_ric + increment_ric
            increment_g = fade_rate * initial_difference_growth
            initial_sales_growth = initial_sales_growth + increment_g * direction_g
            initial_difference_ric = wacc - initial_ric
            initial_difference_growth = initial_sales_growth - lgr
            initial_date = initial_date + datetime.timedelta(days=365)
            g_list.append(initial_sales_growth)
            ric_list.append(initial_ric)
            date_list.append(initial_date)
        ric = pd.Series(ric_list, index=date_list)
        growth = pd.Series(g_list, index=date_list)
        new_nopats = []
        new_deltanoas = []
        discount_rates = []
        for i in range(len(date_list)):
            date = date_list[i]
            new_nopat = initial_nopat + initial_deltanoa * ric.get(date)
            new_deltanoa = initial_nopat * growth.get(date) / ric.get(date)
            new_discount_rate = (1 / (1 + wacc) ** (5 + i))
            new_nopats.append(new_nopat)
            new_deltanoas.append(new_deltanoa)
            discount_rates.append(new_discount_rate)
            initial_nopat = new_nopat
            initial_deltanoa = new_deltanoa
        new_nopats = pd.Series(new_nopats, index=date_list)
        new_deltanoas = pd.Series(new_deltanoas, index=date_list)
        fcffs = new_nopats - new_deltanoas
        discount_rates = pd.Series(discount_rates, index=date_list)
        fcffs_real = fcffs * discount_rates
        return pd.DataFrame({'Sales Growth': growth
                                , 'RIC': ric
                                , 'NOPAT': new_nopats
                                , 'Delta NOA': new_deltanoas
                                , 'FCFFS (Nominal)': fcffs
                                , 'Discount Rates': discount_rates
                                , 'FCFFs (Real)': fcffs_real})

    def calc_terminal_value(self, end_fcff, start_date, end_date, lgr, wacc):
        nominal_term = end_fcff / lgr
        power = end_date.year - start_date.year
        return (1 / (1 + wacc) ** power) * nominal_term

    def get_ov(self):
        explicit_fcff = self.explicit_period['FCFFs (Real)']
        fade_fcff = self.fade_period['FCFFs (Real)']
        total_fcff = explicit_fcff.append(fade_fcff)
        return total_fcff.sum() + self.terminal_value
