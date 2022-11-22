# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:05:58 2022

@author: sigma
"""

# Import pandas and yfinance
import pandas as pd
import yfinance as yf

# Import numpy
import numpy as np
from numpy import *
from numpy.linalg import multi_dot

# Plot settings
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = 16, 8


# Portfolio stocklist
symbols = ['AAPL', 'BA', 'JPM', 'PG', 'CAT', 'XOM', 'AMAT', 'WMT', 'CRM', 'MRK' ]

# Number of assets
numofasset = len(symbols)

# Number of portfolio for optimization
numofportfolio = 5000
# Fetch data from yahoo finance for last six years
stocks_port = yf.download(symbols, start='2019-11-21', end='2022-11-21', progress=False)['Adj Close']

# Verify the output
stocks_port.tail()

#Retrive data
# Let's save the data for future use
stocks_port.to_csv('D:/Python for Quant Model/stocks_port.csv')

# Load locally stored data
df = pd.read_csv('D:/Python for Quant Model/stocks_port.csv', index_col=0, parse_dates=True)

# Check first 5 values 
df.head()

#Descriptive statistics
summary = df.describe().T
summary

# Visualize the data
fig = plt.figure(figsize=(16,8))
ax = plt.axes()

ax.set_title('Normalized Price Plot')
ax.plot(df[-252:]/df.iloc[-252] * 100)
ax.legend(df.columns, loc='upper left')
ax.grid(True)

#Calculate returns
returns = df.pct_change().fillna(0)
returns.head()

# Calculate annual returns
annual_returns = (returns.mean() * 252)
annual_returns

# Visualize the data
fig = plt.figure()
ax =plt.axes()

ax.bar(annual_returns.index, annual_returns*100, color='royalblue', alpha=0.75)
ax.set_title('Annualized Returns (in %)');


#Volatility of each stock return
vols = returns.std()
vols
# Calculate annualized volatilities
annual_vols = vols*sqrt(252)
annual_vols

# Visualize the data
fig = plt.figure()
ax = plt.axes()

ax.bar(annual_vols.index, annual_vols*100, color='orange', alpha=0.5)
ax.set_title('Annualized Volatility (in %)');


#Portfolio statistics
wts = numofasset * [1./numofasset]
array(wts).shape

wts = numofasset * [1./numofasset]
wts = array(wts)[:,newaxis]
wts
wts.shape

#Portfolio return
ret = array(returns.mean() * 252)[:,newaxis]      
ret
ret.shape 
# Portfolio returns
wts.T @ ret     
#Portfolio return
# Covariance matrix
cov = returns.cov() * 252
cov
# Portfolio variance
var = multi_dot([wts.T, cov, wts])
var
# Portfolio volatility
sqrt(var)

#Efficient Frontier
#Constrained optimization
# Import optimization module from scipy
import scipy.optimize as sco

def portfolio_stats(weights):
    
    weights = array(weights)[:,newaxis]
    port_rets = weights.T @ array(returns.mean() * 252)[:,newaxis]    
    port_vols = sqrt(multi_dot([weights.T, returns.cov() * 252, weights])) 
    
    return np.array([port_rets, port_vols, port_rets/port_vols]).flatten()


# Maximizing sharpe ratio
def min_sharpe_ratio(weights):
    return -portfolio_stats(weights)[2]

# Each asset boundary ranges from 0 to 1
tuple((0, 1) for x in range(numofasset))
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(numofasset))
initial_wts = numofasset*[1./numofasset]

# Optimizing for maximum sharpe ratio
opt_sharpe = sco.minimize(min_sharpe_ratio, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)
opt_sharpe
# Portfolio weights
list(zip(symbols,np.around(opt_sharpe['x']*100,2)))


# Portfolio stats
stats = ['Returns', 'Volatility', 'Sharpe Ratio']
list(zip(stats,np.around(portfolio_stats(opt_sharpe['x']),4)))


#Minimum variance portfolio
# Minimize the variance
def min_variance(weights):
    return portfolio_stats(weights)[1]**2

# Optimizing for minimum variance
opt_var = sco.minimize(min_variance, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)
opt_var
# Portfolio weights
list(zip(symbols,np.around(opt_var['x']*100,2)))
# Portfolio stats
list(zip(stats,np.around(portfolio_stats(opt_var['x']),4)))


#Efficient Frontier portfolio
# Minimize the volatility
def min_volatility(weights):
    return portfolio_stats(weights)[1]

targetrets = linspace(0.24,0.48,100)
tvols = []

for tr in targetrets:
    
    ef_cons = ({'type': 'eq', 'fun': lambda x: portfolio_stats(x)[0] - tr},
               {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    opt_ef = sco.minimize(min_volatility, initial_wts, method='SLSQP', bounds=bnds, constraints=ef_cons)
    
    tvols.append(opt_ef['fun'])

targetvols = array(tvols)



# Visualize the simulated portfolio for risk and return
fig = plt.figure()
ax = plt.axes()

ax.set_title('Efficient Frontier Portfolio')

# Efficient Frontier
fig.colorbar(ax.scatter(targetvols, targetrets, c=targetrets / targetvols, 
                        marker='x', cmap='RdYlGn', edgecolors='black'), label='Sharpe Ratio') 

# Maximum Sharpe Portfolio
ax.plot(portfolio_stats(opt_sharpe['x'])[1], portfolio_stats(opt_sharpe['x'])[0], 'r*', markersize =15.0)

# Minimum Variance Portfolio
ax.plot(portfolio_stats(opt_var['x'])[1], portfolio_stats(opt_var['x'])[0], 'b*', markersize =15.0)

ax.set_xlabel('Expected Volatility')
ax.set_ylabel('Expected Return')
ax.grid(True)

