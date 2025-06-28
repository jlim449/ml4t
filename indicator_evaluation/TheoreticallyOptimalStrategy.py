import sys
sys.path.append('../')
import marketsim as ms
from util import get_data, plot_data, get_orders_data_file
import pandas as pd
import datetime as dt

start_date = dt.datetime(2008, 1, 1)
end_date = dt.datetime(2009, 12, 31)


sv = 100000
orders = get_data(['JPM'], pd.date_range(start=start_date, end=end_date), addSPY=False)
orders.dropna(inplace=True)

orders['future_price'] = orders['JPM'].shift(-1)
orders['trade_shares'] = 0
prev_pos = 0


for idx, row in orders.iterrows():
    if row['future_price'] > row['JPM']:
        orders.loc[idx, 'position'] = 1000
    #     if current price is lower than future price
    # short the stock
    elif row['future_price'] < row['JPM']:
        orders.loc[idx, 'position'] = -1000
    else:
        orders.loc[idx, 'position'] = 0



    orders.loc[idx, 'trade_shares'] = orders.loc[idx, 'position'] - prev_pos
    prev_pos = orders.loc[idx, 'position']



#   if previous position includes short position > Buy share


orders

