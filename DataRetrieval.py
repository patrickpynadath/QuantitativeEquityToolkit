import requests
import lxml
from lxml import html
import pandas as pd
import pandas_datareader as pdr
import datetime
import pickle
from company_toolkit import *

def get_sp500_table():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    page = requests.get(url)

    tree = html.fromstring(page.content)
    table = tree.xpath('//table')
    tstring = lxml.etree.tostring(table[0], method='html')
    df = pd.read_html(tstring)[0]
    return df


def get_sp1000_table():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_1000_companies'
    page = requests.get(url)
    tree = html.fromstring(page.content)
    table = tree.xpath('//table')
    tstring = lxml.etree.tostring(table[3], method='html')
    df = pd.read_html(tstring)[0]
    return df

def get_sp500_tickers(sp_table):
    tickers = []
    for i in range(sp_table.shape[0]):
        ticker = sp_table.iat[i, 0]
        if '.' in ticker:
            tickers.append(ticker.replace('.', '-'))
        else:
            tickers.append(ticker)
    return tickers


def get_sp1000_tickers(sp_table):
    tickers = []
    for i in range(sp_table.shape[0]):
        ticker = sp_table.iat[i, 1]
        if '.' in ticker:
            tickers.append(ticker.replace('.', '-'))
        else:
            tickers.append(ticker)
    return tickers

# there are a couple bugs with retrieving this data, will work on later
def save_df_sp1000():
    sp1000_tickers = get_sp1000_tickers(get_sp1000_table())
    start = datetime.date(day=1, month=1, year=2014)
    end = datetime.date.today()
    sp1000_dataframe = pdr.DataReader(sp1000_tickers, 'yahoo', start, end)
    pickle.dump(sp1000_dataframe, open('sp1000_dataframe', 'wb'))
    print("Pickled sp1000")


def get_df(name):
    return pickle.load(open(name, 'rb'))


def get_avail_tickers(dataframe):
    list_avail_tickers = []
    for pair in list(dataframe.columns):
        ticker = pair[1]
        if ticker not in list_avail_tickers:
            list_avail_tickers.append(ticker)
    return list_avail_tickers


def update_dataframe_index(original_dataframe):
    last_avail_date = list(original_dataframe.index)[-1]
    print(last_avail_date)
    current_date = datetime.date.today()
    print(datetime.date.today())
    if current_date != last_avail_date:
        if current_date - last_avail_date.date() > datetime.timedelta(days=2):
            print("Stock Dataframe not up to date, will update immediately")
            tickers = get_avail_tickers(original_dataframe)
            df = pdr.DataReader(tickers, 'yahoo', last_avail_date, current_date).drop(index=last_avail_date)
            print('Retrieved new data')
            new_dataframe = original_dataframe.append(df)
            return new_dataframe
            #pickle.dump(new_dataframe, open('stock_dataframe.pickle', 'wb'))
            print("Pickled updated stock dataframe")
        else:
            print("No new data")
    else:
        print("No new data")


def update_dataframe_ticker(original_dataframe, tickers):
    import company_toolkit
    print('Updating stock_dataframe with ' + str(tickers))
    start = list(original_dataframe.index)[0]
    end = list(original_dataframe.index)[-1]
    try:
        df = pdr.DataReader(tickers, 'yahoo', start, end)
        new_dataframe = pd.concat([original_dataframe, df], axis=1)
        pickle.dump(new_dataframe, open('stock_dataframe.pickle', 'wb'))
        print('Pickled updated dataframe, now updating company_dictionary')
        company_dict = get_company_dict()
        for ticker in tickers:
            if ticker != '^GSPC':
                c = Company(ticker)
                company_dict.update({ticker : c})
        print("Updated Company Dictionary")
        dump_company_dict(company_dict)
        return new_dataframe
    except KeyError:
        print('There is no data on ' + str(ticker) + ' in yahoo finance')


