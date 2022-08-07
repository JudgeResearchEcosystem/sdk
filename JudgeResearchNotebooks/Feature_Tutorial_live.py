

### 1.  pass your keys ###
JR_API_KEY = 'xxx' 
CA_API_KEY = 'xxx'

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

def percDif(y):
    x = (y['Open'] - y['Close']) / y['Open']
    z = pd.concat(([y['StartDate'], x]), axis = 1)
    z.columns = ['StartDate', 'feature']
    z = z.applymap(str)
    return(z)


### 4.  finally, use live_fcs (fetch, calculate & submit) ###
### The first row of arguments structures the delays      ###
### the next row structures the data-fetching arguments   ###
### the third is the function that engineers your feature(s)#
### and the last gives information you pass to the JR API ###

myDVs = ('ETH-USD', 'V-ETH-USD', 'BTC-USD', 'V-BTC-USD') # j as counter
myVarNames = ('BTCt', 'ETHt', 'SOLt')
lv = len(myVarNames)

exchList1 = ['BINANCE', 'BINANCE', 'BINANCE']
assetList1 = ['BTC-USDT-SPOT', 'ETH-USDT-SPOT', 'SOL-USDT-SPOT']
ippTuple = ('last', 'last', 'last')
mbsTuple = ('45', '45', '45')

for j in range(len(myDVs)):
    thisDV = (myDVs[j], ) * lv
    jh.live_fcs(firstdelay = .00000001, seconddelay = .00000002,
                timeBloc = '45m', nObsBack = 0, APIKey = CA_API_KEY, 
                exchangeList = exchList1, assetList = assetList1,
                thisFunc = percDif, 
                JRAPIKEY = JR_API_KEY, featureNames = myVarNames, ippProc = ippTuple, DV = thisDV, MBS = mbsTuple )

### see the JudgeHelperFuncs section 2 for more info      ###
### on fcs and its wrapper function, live_fcs             ###
