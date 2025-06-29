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
    order_file : np.ndarray,
    start_val=100000,
    start_date = dt.datetime(2008, 1, 1),
    end_date = dt.datetime(2009, 12, 31),
    commission=0,
    impact=0,
):
    start_date = order_file.index[0]
    end_date = order_file.index[-1]
    symbol = order_file.columns[0]
    prices = get_data([symbol], pd.date_range(start_date, end_date), addSPY=False)

    # consolidated = [get_data([i], pd.date_range(start_date, end_date)) for i in unique_sym]
    df = order_file.copy()
    prices.dropna(inplace=True)
    prices['cash'] = 1.0

    # Create trades DataFrame
    trades = prices.copy()
    trades.loc[:, :] = 0.0
    # holding
    holding = trades.copy()
    holding.iloc[0, holding.columns.get_loc('cash')] = start_val
    # holding = trades.copy()

    # trades.iloc[0, trades.columns.get_loc('cash')] = 1000000.0
    # caseh
    for index, row in df.iterrows():
        dt = index
        sym = row.index[0]
        share = row[symbol]

        trades.loc[dt, sym] += share
        current_price = prices.loc[dt, sym]
        trades.loc[dt, 'cash'] -= share * current_price

    holding = trades.cumsum()
    holding['cash'] += start_val
    stocks = prices *  holding
    port_val = stocks.sum(axis = 1)
    port_val = pd.DataFrame(port_val)
    port_val.columns = ['Total_Value']

    return port_val
  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		 	 	 			  		 			 	 	 		 		 	
    test_code()  		  	   		 	 	 			  		 			 	 	 		 		 	
