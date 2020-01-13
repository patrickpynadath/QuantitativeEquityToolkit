#!/Users/anaconda3/bin/python

import DataRetrieval
import pandas as pd
import pandas_datareader
import pickle
import datetime

original_dataframe = pickle.load(open('/Users/patrickpynadath/PycharmProjects/QuantitativeToolkit/stock_dataframe.pickle', 'rb'))
if datetime.date.today() != original_dataframe.index[-1].date():
    if datetime.date.today() - original_dataframe.index[-1].date() > datetime.timedelta(days=2):
        DataRetrieval.update_dataframe_index(original_dataframe)
