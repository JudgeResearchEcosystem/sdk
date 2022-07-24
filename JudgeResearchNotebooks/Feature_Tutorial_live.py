JR_API_KEY = "hEkFs9wE1ud5qIqvDH2A2JPBvf8xkDb9bAGmSf30" # Your Judge Research API was given to you upon sign-up.  You can find it under your profile.
CA_API_KEY = "<your API key here>" # Sign up at coinalytix.io

# Import classes that handle API connections and parameters
from historical_data import Coinalytix, HDParams
from judgeresearch import JudgeResearch, JRParams

# Import classes for data handling & visualization
import json
import scipy 
import plotly.graph_objects as go
import pandas as pd
from datetime import date, datetime, timedelta
import pandas_ta as ta
import time
import math
import numpy as np
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

import JudgeHelperFuncs as jh



from watchlist import colors

### 2. to figure out what to tell coinalytix to call:

def beginOfPeriod(thisTime = datetime.now(), roundTo = 45):
    rounded = thisTime - (thisTime - datetime.min) % timedelta(minutes=roundTo)
    rounded = rounded.strftime("%Y-%m-%d %H:%M:%S")
    return rounded

# to give to fcs to fetch coinalytix data:

def define_asset_For_HDParams(periodsPerDay=32, interval = 45, intervalUnit = 'm', nObs = 1,
                          exchange = 'BINANCE', ticker = "BTC-USD-SPOT"):
    startDateString = beginOfPeriod(thisTime = datetime.now(), roundTo = interval) #     
    startDate = datetime.strptime(startDateString, "%Y-%m-%d %H:%M:%S")
    timeBack = datetime.now() - startDate #      

    asset = HDParams() #                          Set exchange, must be "BINANCE" or ...
    asset.exchange = exchange #                   Set asset, currently supports "BTC-USD-SPOT", "ETH-USD-SPOT", ...
    asset.ticker = ticker #               
    asset.set_start_date(startDateString) #       
    asset.interval = str(interval) + intervalUnit # Example arguments:  45m 4h, 1d                     
    asset.num_periods = nObs #                    Set number of reporting periods
    return(asset)
    
### actual feature generation function: ###
    
def feature_gen(hddf):
    y1 = (hddf['Open'] - hddf['Close']) / hddf['Open']
    y1 = y1 ** 3
    y1 = pd.concat(([hddf['StartDate'], y1]), axis = 1)
    y1.columns = ['StartDate', 'feature']
    y1 = y1.applymap(str)
    return(y1)

### combines fetching data, feature generation and submission functions ###

def fcs(asset, params):
    """ fcs:  a simple wrapper function to fetch, calculate, and submit data to Judge Research. 
    This function is meant to slightly shorten your live submission scripts
    It takes in (1) the "asset" parameter from Step 2 of the tutorial that comes up when you fist open the JR SDK,
    and (2) the "params" parameter that tells the JR API what feature you are submitting (e.g. in Step 5 of the tutorial)  
    And it returns the response from the JR API
    """
    asset = define_asset_For_HDParams(periodsPerDay=32, interval = 45, intervalUnit = 'm', nObs = 1,
                          exchange = 'BINANCE', ticker = "BTC-USD-SPOT")
    hddf = HD.fetch_hd(asset) #                 Get your hddf: historical data dataframe
    fdf = feature_gen(hddf) #                   Call your function & get your fdf: feature dataframe
    features = JR.craft_features(params, fdf) # Craft your features from the dataframe 
    payload = JR.format_payload(features) #     Format your payload
    submit = JR.submit_feature(payload) #       And send it in
    return submit
    
### calls fcs three times ###    
    
def live_fcs(asset, params, firstdelay, seconddelay):
        """ live_fcs:  
        calculate and submit feature data 3 times, throttled by percentage
        of time remaining until end of widow
        
        first submission = immediate
        second submission = % of remaining time in window ex. .30 = 3 minutes of 10 remaining
        last submission = % of remaining time in window ex, .90 = 9 of 10 remaining
        """
        
        ##### Sec. 1:  Detect current time window & calculate times #####
        now = datetime.now() #                                         1. determine current time
        print("Current time: " + now.strftime("%Y-%m-%dT%H:%M:%SZ"))
        
        start_of_day = datetime.utcnow().today().replace(microsecond=0, second=0, minute=0, hour=0) # 2. determine the upcoming deadline (end of window) # TO BE IMPROVED AFTER ALPHA TEST: don't assume windows are anchored to 00:00

        deadline = start_of_day #                                      3. set deadline anchor time
        print("Start of today:" + start_of_day.strftime("%Y-%m-%dT%H:%M:%SZ"))
        
        window = timedelta(minutes=int(params.mbs)) #                  4. create timedelta object based on MBS parameter
        
        while now > deadline: #                                        5. iterate through block times until current time is passed, set deadline
            deadline = deadline + window
            
        print("Next Block End: " + deadline.strftime("%Y-%m-%dT%H:%M:%SZ"))
        
        remaining = deadline - now #                                   6. calculate time remaining until current window closes
        delay1 = int(remaining.total_seconds() * firstdelay) #         7. calculate delays (seconds)
        delay2 = int(remaining.total_seconds() * seconddelay) - delay1
        
        ##### Sec. 2:  use fcs (fetch, calculate & submit) to send in features three times #####
        s1 = fcs(asset, ft_params) #                                   1. immediately
        print("First submission at " + datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"))
        print(s1)
        
        time.sleep(delay1) #                                           2. sleep, & run fcs after delay 1
        s2 = fcs(asset, ft_params)
        print("Second submission at " + datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"))
        print(s2)
        
        time.sleep(delay2) #                                           3. and a final time.
        s3 = fcs(asset, ft_params)
        print("Final submission at " + datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"))        
        print(s3)
        

live_fcs(asset, ft_params, .0050, .0090)