

### 1.  pass your keys ###
JR_API_KEY = "<your api key here>" 
CA_API_KEY = '<your api key here>'

### 2. load the workspace ###
from historical_data import Coinalytix, HDParams
from judgeresearch import JudgeResearch, JRParams
import JudgeHelperFuncs as jh

import json ### remove unnecessary package imports below 
import scipy 
import plotly.graph_objects as go
import plotly
import pandas as pd
from datetime import date, datetime, timedelta
import pandas_ta as ta
import time
import math
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf 
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
import sklearn
import requests # these two not in other imports
import json
from watchlist import colors

### 3. The function you are using to generate features:   ###
### in this case the simplest possible feature            ###  

def perc_dif(hddf):
    y = (hddf['Open'] - hddf['Close']) / hddf['Open']
    y = y 
    y = pd.concat(([hddf['StartDate'], y]), axis = 1)    ### note the three formatting lines 
    y.columns = ['StartDate', 'feature']
    y = y.applymap(str)
    return y

### 4.  finally, use live_fcs (fetch, calculate & submit) ###
### The first row of arguments structures the delays      ###
### the next row structures the data feteching arguments  ###
### the third is the function that engineers your feature(s)#
### and the last gives information you pass to the JR API ###

jh.live_fcs(firstdelay = .005, seconddelay = .01,
            timeBloc = '45m', nObsBack = 0, APIKey = CA_API_KEY, exchangeList = ['BINANCE', 'BINANCE'], assetList = ['BTC-USDT-SPOT', 'ETH-USDT-SPOT'],
            thisFunc = perc_dif, 
            JRAPIKEY = JR_API_KEY, featureNames = ('BTCt', 'ETHt'), ippProc = ('last', 'last'), DV = ('BTC-USD', 'BTC-USD'), MBS = ('45', '45') )

### see the JudgeHelperFuncs section 2 for more info      ###
### on fcs and its wrapper function, live_fcs             ###