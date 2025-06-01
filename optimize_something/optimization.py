""""""  		  	   		 	 	 			  		 			 	 	 		 		 	
"""MC1-P2: Optimize a portfolio.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
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
  		  	   		 	 	 			  		 			 	 	 		 		 	
Student Name: Jae Hwan Lim (replace with your name)  		  	   		 	 	 			  		 			 	 	 		 		 	
GT User ID: jlim449 (replace with your User ID)  		  	   		 	 	 			  		 			 	 	 		 		 	
GT ID: 900897987 (replace with your GT ID)  		  	   		 	 	 			  		 			 	 	 		 		 	
"""  		  	   		 	 	 			  		 			 	 	 		 		 	
import sys
sys.path.append('../')
  		  	   		 	 	 			  		 			 	 	 		 		 	
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt  		  	   		 	 	 			  		 			 	 	 		 		 	
import pandas as pd  		  	   		 	 	 			  		 			 	 	 		 		 	
from util import get_data, plot_data
import scipy.optimize as opt
  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
# This is the function that will be tested by the autograder  		  	   		 	 	 			  		 			 	 	 		 		 	
# The student must update this code to properly implement the functionality  		  	   		 	 	 			  		 			 	 	 		 		 	



def sharpe_ratio(allocs, prices):
    norm_prices = prices / prices.iloc[0]
    allocated = norm_prices * allocs
    port_val = allocated.sum(axis=1)
    daily_returns = port_val.pct_change()
    daily_returns.dropna(inplace=True)
    adr = daily_returns.mean()
    sddr = daily_returns.std()
    return -adr / sddr * np.sqrt(252)



def optimize_portfolio(  		  	   		 	 	 			  		 			 	 	 		 		 	
    sd=dt.datetime(2008, 1, 1),  		  	   		 	 	 			  		 			 	 	 		 		 	
    ed=dt.datetime(2009, 1, 1),  		  	   		 	 	 			  		 			 	 	 		 		 	
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		 	 	 			  		 			 	 	 		 		 	
    gen_plot=False,  		  	   		 	 	 			  		 			 	 	 		 		 	
):  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		 	 	 			  		 			 	 	 		 		 	
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		 	 	 			  		 			 	 	 		 		 	
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		 	 	 			  		 			 	 	 		 		 	
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		 	 	 			  		 			 	 	 		 		 	
    statistics.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type sd: datetime  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type ed: datetime  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		 	 	 			  		 			 	 	 		 		 	
        symbol in the data directory)  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type syms: list  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		 	 	 			  		 			 	 	 		 		 	
        code with gen_plot = False.  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type gen_plot: bool  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		 	 	 			  		 			 	 	 		 		 	
        standard deviation of daily returns, and Sharpe ratio  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: tuple  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Read in adjusted closing prices for given symbols, date range  		  	   		 	 	 			  		 			 	 	 		 		 	
    dates = pd.date_range(sd, ed)  		  	   		 	 	 			  		 			 	 	 		 		 	
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		 	 	 			  		 			 	 	 		 		 	
    prices = prices_all[syms]  # only portfolio symbols  		  	   		 	 	 			  		 			 	 	 		 		 	
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # find the allocations for the optimal portfolio  		  	   		 	 	 			  		 			 	 	 		 		 	
    # note that the values here ARE NOT meant to be correct for a test case  		  	   		 	 	 			  		 			 	 	 		 		 	

    random_weights = np.random.random(len(syms))
    allocs =  random_weights / np.sum(random_weights)  # normalize to sum to 1

    allocs = np.asarray(allocs)  # convert to numpy array
 	   		 	 	 			  		 			 	 	 		 		 	
    cr, adr, sddr, sr = [  		  	   		 	 	 			  		 			 	 	 		 		 	
        0.25,  		  	   		 	 	 			  		 			 	 	 		 		 	
        0.001,  		  	   		 	 	 			  		 			 	 	 		 		 	
        0.0005,  		  	   		 	 	 			  		 			 	 	 		 		 	
        2.1,  		  	   		 	 	 			  		 			 	 	 		 		 	
    ]  # add code here to compute stats  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Get daily portfolio value  		  	   		 	 	 			  		 			 	 	 		 		 	
    # port_val = prices_SPY  # add code here to compute daily portfolio values
    # prices * allocs


    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = [(0.0, 1.0)] * len(syms)

    result = opt.minimize(sharpe_ratio, allocs, args=(prices,),  method='SLSQP', bounds=bounds, constraints=cons, options={'disp': True})


    allocs = result.x
    norm_prices = prices / prices.iloc[0]
    allocated = norm_prices * allocs
    port_val = allocated.sum(axis=1)
    daily_returns = port_val.pct_change()
    daily_returns.dropna(inplace=True)
    adr = daily_returns.mean()
    sddr = daily_returns.std()
    sr = -sharpe_ratio(allocs, prices)


    # Compare daily portfolio value with SPY using a normalized plot  		  	   		 	 	 			  		 			 	 	 		 		 	
    if gen_plot:  		  	   		 	 	 			  		 			 	 	 		 		 	
        # add code to plot here  		  	   		 	 	 			  		 			 	 	 		 		 	

        normalized_SPY = prices_SPY / prices_SPY.iloc[0]

        df_temp = pd.concat(
            [port_val, normalized_SPY], keys=["Portfolio", "SPY"], axis=1
        )
        plt.figure(figsize=(10, 6))
        plt.title('Normalized Return : Optimal Portfolio vs SPY')
        plt.plot(df_temp.index, df_temp["Portfolio"], label="Portfolio")
        plt.plot(df_temp.index, df_temp["SPY"], label="SPY")
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Nomralized Value")
        plt.savefig('Figure1.png')
        pass
  		  	   		 	 	 			  		 			 	 	 		 		 	
    return np.round(allocs, 2), cr, adr, sddr, sr
  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
def test_code():  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    This function WILL NOT be called by the auto grader.  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	

    # start_date = str2dt("2011-01-01"),
    # end_date = str2dt("2011-12-31"),
    # symbols = ["AAPL", "GLD", "GOOG", "XOM"]

    start_date = dt.datetime(2011, 1, 1)
    end_date = dt.datetime(2011, 12, 31)
    symbols = ["AAPL", "GLD", "GOOG", "XOM"]
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Assess the portfolio  		  	   		 	 	 			  		 			 	 	 		 		 	
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		 	 	 			  		 			 	 	 		 		 	
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Print statistics  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Start Date: {start_date}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"End Date: {end_date}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Symbols: {symbols}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Allocations:{allocations}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Sharpe Ratio: {sr}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Average Daily Return: {adr}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Cumulative Return: {cr}")  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		 	 	 			  		 			 	 	 		 		 	
    # This code WILL NOT be called by the auto grader  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Do not assume that it will be called  		  	   		 	 	 			  		 			 	 	 		 		 	
    test_code()  		  	   		 	 	 			  		 			 	 	 		 		 	
