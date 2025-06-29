""""""  		  	   		 	 	 			  		 			 	 	 		 		 	
"""MC2-P1: Market simulator.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			 	 	 		 		 	
or edited.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Student Name: Jae Hwan Lim  		  	   		 	 	 			  		 			 	 	 		 		 	
GT User ID: jlim443 (replace with your User ID)  		  	   		 	 	 			  		 			 	 	 		 		 	
GT ID: 900897987 (replace with your GT ID)  		  	   		 	 	 			  		 			 	 	 		 		 	
"""  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
import datetime as dt  		  	   		 	 	 			  		 			 	 	 		 		 	
import os
import sys
sys.path.append('../')
  		  	   		 	 	 			  		 			 	 	 		 		 	
import numpy as np  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
import pandas as pd  		  	   		 	 	 			  		 			 	 	 		 		 	
from util import get_data, plot_data, get_orders_data_file

def author():
    return 'jlim443'


def gtid():
    """
    :return: The GT ID of the student
    :rtype: int
    """
    return 900897987  # replace with your GT ID number


def compute_portvals(
    orders_file="./orders/orders.csv",  		  	   		 	 	 			  		 			 	 	 		 		 	
    start_val=1000000,  		  	   		 	 	 			  		 			 	 	 		 		 	
    commission=9.95,  		  	   		 	 	 			  		 			 	 	 		 		 	
    impact=0.005,  		  	   		 	 	 			  		 			 	 	 		 		 	
):  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    Computes the portfolio values.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param orders_file: Path of the order file or the file object  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type orders_file: str or file object  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param start_val: The starting value of the portfolio  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type start_val: int  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type commission: float  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type impact: float  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: pandas.DataFrame  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    # this is the function the autograder will call to test your code  		  	   		 	 	 			  		 			 	 	 		 		 	
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		 	 	 			  		 			 	 	 		 		 	
    # code should work correctly with either input  		  	   		 	 	 			  		 			 	 	 		 		 	
    # TODO: Your code here  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # In the template, instead of computing the value of the portfolio, we just  		  	   		 	 	 			  		 			 	 	 		 		 	
    # read in the value of IBM over 6 months

    df = pd.read_csv(orders_file)
    unique_sym = np.unique(df.Symbol).tolist()

    start_date = pd.to_datetime(df.Date.min())
    end_date = pd.to_datetime(df.Date.max())



    # consolidated = [get_data([i], pd.date_range(start_date, end_date)) for i in unique_sym]
    consolidated = get_data(unique_sym, pd.date_range(start_date, end_date))
    # remove spy
    consolidated = consolidated[unique_sym]
    consolidated['cash'] = 1

    # keep track of trades
    trades = consolidated.copy()
    trades.loc[:, trades.columns] = 0
    trades['cash'] = 0

    # holding
    # holding = trades.copy()
    # holding.iloc[0, holding.columns.get_loc('cash')] = start_val
    # holding = trades.copy()

    # trades.iloc[0, trades.columns.get_loc('cash')] = 1000000.0
    # caseh
    for index, row in df.iterrows():
        dt = row.Date
        sym = row.Symbol
        order = row.Order
        if order == 'BUY':
            share = row.Shares
        else:
            share = - row.Shares

        trades.loc[dt, sym] += share
    #     get the price & compute total share
    #     total_price = 0
        current_price = consolidated.loc[dt, sym]
        # calculate cash holding

        impact_fee = abs(share * current_price) * impact

        # trades = consolidated * consolidate_copy
        trades.loc[dt, 'cash'] -= share * current_price
        trades.loc[dt, 'cash'] -= commission
        trades.loc[dt, 'cash'] -= impact_fee


    holding = trades.cumsum()
    holding['cash'] = holding['cash'] + start_val
    stocks = holding[unique_sym] * consolidated[unique_sym]
    port_val = stocks.sum(axis = 1) + holding['cash']
    port_val = pd.DataFrame(port_val)
    port_val.columns = ['Total_Value']

    return port_val
  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
def test_code():  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    Helper function to test code  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    # this is a helper function you can use to test your code  		  	   		 	 	 			  		 			 	 	 		 		 	
    # note that during autograding his function will not be called.  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Define input parameters  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    of = "./orders/orders-02.csv"
    sv = 1000000  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Process orders  		  	   		 	 	 			  		 			 	 	 		 		 	
    portvals = compute_portvals(orders_file=of, start_val=sv, commission=0, impact = 0)
    if isinstance(portvals, pd.DataFrame):  		  	   		 	 	 			  		 			 	 	 		 		 	
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		 	 	 			  		 			 	 	 		 		 	
    else:  		  	   		 	 	 			  		 			 	 	 		 		 	
        "warning, code did not return a DataFrame"  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Get portfolio stats

    def sharpe_ratio(port_val):
        daily_returns = port_val.pct_change()
        daily_returns.dropna(inplace=True)
        adr = daily_returns.mean()
        sddr = daily_returns.std()
        return adr / sddr * np.sqrt(252)

    daily_returns = portvals.pct_change()
    daily_returns.dropna(inplace=True)


    cr = (portvals.iloc[-1] / sv) - 1
    adr = daily_returns.mean()
    sddr = daily_returns.std()
    sr = sharpe_ratio(portvals)



    # Here we just fake the data. you should use your code from previous assignments.
    start_date = portvals.index[0]
    end_date = portvals.index[-1]

    spy = get_data(['SPY'], pd.date_range(start_date, end_date))
    spy_cr = (spy.iloc[-1] / spy.iloc[0]) - 1

    spy_daily_returns = spy.pct_change()
    spy_daily_returns.dropna(inplace=True)

    spy_adr = spy_daily_returns.mean()
    spy_sddr = spy_daily_returns.std()
    spy_sr = sharpe_ratio(spy)


    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [  		  	   		 	 	 			  		 			 	 	 		 		 	
        cr,
        adr,
        sddr,
        sr,
    ]  		  	   		 	 	 			  		 			 	 	 		 		 	
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [  		  	   		 	 	 			  		 			 	 	 		 		 	
        spy_cr[0],
        spy_adr[0],
        spy_sddr[0],
        spy_sr[0]
    ]  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Compare portfolio against $SPX  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Date Range: {start_date} to {end_date}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print()  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print()  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print()  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print()  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print()  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Final Portfolio Value: {portvals[-1]}")  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		 	 	 			  		 			 	 	 		 		 	
    test_code()  		  	   		 	 	 			  		 			 	 	 		 		 	
