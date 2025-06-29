import sys
sys.path.append('../')
import marketsim as ms
from util import get_data, plot_data, get_orders_data_file
import pandas as pd
import datetime as dt
import numpy as np



def testpolicy(symbol = 'JPM', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)


    # sv = 100000
    df = get_data(['JPM'], pd.date_range(start=start_date, end=end_date), addSPY=False)

    orders = df.copy()

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

    df = orders.copy()
    df = df['trade_shares']
    df = pd.DataFrame(df)


    df.rename(columns={df.columns[0]: symbol}, inplace=True)

    return df

df_trades = testpolicy('JPM')


#   if previous position includes short position > Buy share

portvals = ms.compute_portvals(
        order_file=df_trades,  # Pass your trades DataFrame
    )


def sharpe_ratio(port_val):
    daily_returns = port_val.pct_change()
    daily_returns.dropna(inplace=True)
    adr = daily_returns.mean()
    sddr = daily_returns.std()
    return adr / sddr * np.sqrt(252)


def portfolio_performance(portvals, sv):

    daily_returns = portvals.pct_change()
    daily_returns.dropna(inplace=True)

    cr = (portvals.iloc[-1] / sv) - 1
    adr = daily_returns.mean()
    sddr = daily_returns.std()
    sr = sharpe_ratio(portvals)

    cum_ret, avg_daily_ret, std_daily_ret, sr = [
        cr,
        adr,
        sddr,
        sr,
    ]


    return cum_ret, avg_daily_ret, std_daily_ret, sr


cum_ret, avg_daily_ret, std_daily_ret, sr = portfolio_performance(portvals, 100000)


cum_ret, avg_daily_ret, std_daily_ret, sr



