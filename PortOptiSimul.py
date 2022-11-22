# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:28:55 2022

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

# Load locally stored data
df = pd.read_csv('D:/Python for Quant Model/stocks_port.csv', index_col=0, parse_dates=True)
#Calculate returns
returns = df.pct_change().fillna(0)


#Portfolio simulation#############


#Port
w = random.random(numofasset)[:, newaxis]
w

# Set weights such that sum of weights equals 1
w /= sum(w)
w
w.shape, sum(w)


# Initialize the lists
rets = []; vols = []; wts = []

# Simulate 5,000 portfolios
for i in range (5000):
    
    # Generate random weights
    weights = random.random(numofasset)[:, newaxis]
    
    # Set weights such that sum of weights equals 1
    weights /= sum(weights)
    
    # Portfolio statistics
    rets.append(weights.T @ array(returns.mean() * 252)[:, newaxis])        
    vols.append(sqrt(multi_dot([weights.T, returns.cov()*252, weights])))
    wts.append(weights.flatten())

# Record values     
port_rets = array(rets).flatten()
port_vols = array(vols).flatten()
port_wts = array(wts)

port_rets
port_vols
port_wts
port_rets.shape, port_vols.shape, port_wts.shape

# Create a dataframe for analysis
msrp_df = pd.DataFrame({'returns': port_rets,
                      'volatility': port_vols,
                      'sharpe_ratio': port_rets/port_vols,
                      'weights': list(port_wts)})
msrp_df.head()


# Summary Statistics
msrp_df.describe().T


#Maximize sharpe ratio
# Max sharpe ratio portfolio 
msrp = msrp_df.iloc[msrp_df['sharpe_ratio'].idxmax()]
msrp
# Max sharpe ratio portfolio weights
max_sharpe_port_wts = msrp_df['weights'][msrp_df['sharpe_ratio'].idxmax()]

# Allocation to achieve max sharpe ratio portfolio
dict(zip(symbols,np.around(max_sharpe_port_wts*100,2)))


# Visualize the simulated portfolio for risk and return
fig = plt.figure()
ax = plt.axes()

ax.set_title('Monte Carlo Simulated Allocation')

# Simulated portfolios
fig.colorbar(ax.scatter(port_vols, port_rets, c=port_rets / port_vols, 
                        marker='o', cmap='RdYlGn', edgecolors='black'), label='Sharpe Ratio') 

# Maximum sharpe ratio portfolio
ax.scatter(msrp['volatility'], msrp['returns'], c='red', marker='*', s = 300, label='Max Sharpe Ratio')

ax.set_xlabel('Expected Volatility')
ax.set_ylabel('Expected Return')
ax.grid(True)

