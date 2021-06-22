#!/usr/bin/env python
# coding: utf-8

# DA 420 Final Project (Stock Price Prediction LTSM trading bot)

# In[14]:


# importing necessary packages

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import sys  
sys.path.insert(0, r'C:\Users\Taterthot\Desktop\420 final')
import ticker
#import algo

ticker_list = ticker.trending_ticker()
print(ticker_list)


# In[17]:


def main():
    budget = float(input("Please input your investment amount: "))
    
    # stock symbols to look into within price range
    counter = 0
    ticker_list["last_price"] = pd.to_numeric(ticker_list["last_price"])
    trade_list = ticker_list[ticker_list['last_price'] <= budget] 
        
    print(trade_list)
    
    return(trade_list)

