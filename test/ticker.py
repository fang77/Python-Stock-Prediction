#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests

def trending_ticker():
    URL = "https://finance.yahoo.com/trending-tickers"
    dat = requests.get(URL)

    soup = BeautifulSoup(dat.text)
    top_ticker = []
    last_price = []
    ticker_name = []
    volume = []

    for listing in soup.find_all('tr', attrs={'class':'simpTblRow'}):

        for individual in listing.find_all('td', attrs={'aria-label':'Symbol'}):
          top_ticker.append(individual.text)

        for individual in listing.find_all('td', attrs={'aria-label':'Name'}):
          ticker_name.append(individual.text)

        for individual in listing.find_all('td', attrs={'aria-label':'Last Price'}):
          last_price.append(individual.text)

        for individual in listing.find_all('td', attrs={'aria-label':'Volume'}):
          volume.append(individual.text)

    fin_dat = pd.DataFrame()
    
    fin_dat['symbol'] = top_ticker
    fin_dat['company_name'] = ticker_name
    fin_dat['last_price'] = last_price
    fin_dat['volume'] = volume
    
    # pruning
    fin_dat["last_price"] = fin_dat["last_price"].str.replace(',','')
    fin_dat["last_price"] = pd.to_numeric(fin_dat["last_price"])
    fin_dat["volume"] = fin_dat["volume"].str.replace(',','')
    fin_dat["volume"] = fin_dat["volume"].str.replace('M','')
    fin_dat.loc[fin_dat['volume'].str.contains('B'), 'volume'] = '1000'
    
    fin_dat["volume"] = fin_dat["volume"].str.replace('B','')
    fin_dat["volume"] = pd.to_numeric(fin_dat["volume"])
    
    # REMOVE CRYPTO
    fin_dat = fin_dat[~fin_dat["symbol"].str.contains("USD")]
    fin_dat = fin_dat[~fin_dat["symbol"].str.contains("CAD")]
    
    # limit to volume to tickers above 100 million
    fin_dat = fin_dat[fin_dat['volume'] >= 100.0] 
    
    return fin_dat

print(trending_ticker())

