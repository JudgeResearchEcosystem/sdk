# time handline funcions
from datetime import date, datetime, timedelta
from historical_data import Coinalytix, HDParams
from judgeresearch import JudgeResearch, JRParams

# Import classes for data handling & visualization
import json
import requests
import scipy 
import plotly.graph_objects as go
import pandas as pd
from datetime import date, datetime, timedelta
import pandas_ta as ta
import time
import math
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf 
import itertools
from functools import partial

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

### ----------------------------------------------------------------------------------
###
### ----- section 1. is tiny time-saver functions.  
###  i.e. it's silly to specify the axis every time for something as common as: 
### pd.concat([df1, df2], axis=1)
###
### ----- section 2. is for wrapper functions for the SDK.  
### They don't necessarily make the SDK simpler or more functional
### but they largely reduce lines of code so you can make calling 
### data, sending it in, etc, a little quicker, more easily embed it 
### in functions, etc.  
###
### ----- section 3. are general data manipulation functions.  Like section 1., 
### they are largely about making pandas/etc.'s inappropriately verbose (from   
### the perspective of data science) functions enjoy default 
### settings for how researchers use those functions the vast majority of the time
### 
### ----- section 4. Is the same as section 3 but for models & stat. analysis
### as opposed to data manipulation
###
### ----------------------------------------------------------------------------------

### ----- Section 1. Tiny Time-Saver Functions  ------ ###

def cb(x1, x2): #                                      improve so it's any number of cbinds
    x = pd.concat([x1, x2], axis=1)
    return x 

# --- a more parsimonious paste function

def paste(*args, sep = ' ', collapse = None):
    """
    Port of paste from R
    Args:
        *args: lists to be combined
        sep: a string to separate the terms
        collapse: an optional string to separate the results
    Returns:
        A list of combined results or a string of combined results if collapse is not None
    """
    combs = list(itertools.product(*args))
    out = [sep.join(str(j) for j in i) for i in combs]
    if collapse is not None:
        out = collapse.join(out)
    return out

# --- same w/ paste0

paste0 = partial(paste, sep = '')

### ----- Section 2. JR SDK Wrapper Functions   ------ ###

def beginOfPeriod(thisTime = "now", timeBloc = 240, nObsBack = 0):
    """ whatever the current time, gives the time period at the beginning 
    of a period of arbitrary size
    lasteTimeBloc can be integer or string
    """
    if isinstance(timeBloc, str): 
        unitType = timeBloc[-1]                      
        unitSize = int(timeBloc[:-1])
        if unitType == "m":
            timeBloc = unitSize
        elif unitType == "h":
            timeBloc = unitSize * 60 
        elif unitType == "s":
            timeBloc = unitSize / 60
        elif unitType == "d":
            timeBloc = unitSize * 60 * 24                       
    if thisTime == "now":
        thisTime = datetime.now()
    rounded = thisTime - (thisTime - datetime.min) % timedelta(minutes=timeBloc) 
    if nObsBack != 0:
        rounded = rounded - timedelta(minutes=timeBloc*nObsBack) 
    rounded = rounded.strftime("%Y-%m-%d %H:%M:%S")
    return rounded

# ---

def formContempCallWindow(startDateString = "2021-07-01 00:00:00", perSize = "4h", usefloor = True):
    """ helper function saves a few lines when  
    forming a contempoary window to call coinalytix
    NOTE:  Only works with minutes, hours, days & seconds, supplied as m, h, d, or s
    """
    unitType = perSize[-1]                      
    unitSize = int(perSize[:-1])
    if unitType == "m":
        periodsPerDay = (60 * 24) / unitSize
    elif unitType == "h":
        periodsPerDay = 24 / unitSize 
    elif unitType == "s":
        periodsPerDay = (60 * 60 * 24) / unitSize
    elif unitType == "d":
        periodsPerDay = 1 / unitSize                       
    startDate = datetime.strptime(startDateString, "%Y-%m-%d %H:%M:%S")
    timeBack = datetime.now() - startDate
    nObs = timeBack.total_seconds() / ((60*60*24) / periodsPerDay) 
    if usefloor == True:                        
        nObs = math.floor(nObs)
    else:
        nObs = math.ceil(nObs)
    dictToReturn = {
        "startDateString" : startDateString,
        "periodsPerDay" : periodsPerDay,
        "perSize" : perSize,
        "nObs" : nObs}
    return dictToReturn                      

# ---

def define_asset_For_HDParams(exchange = 'BINANCE', ticker = "BTC-USD-SPOT", 
                             startDateString = "2021-07-01 00:00:00", 
                             perSize = '45m', nObs = 1):
    asset = HDParams() #                          Set exchange, must be "BINANCE" or ...
    asset.exchange = exchange #                   Set asset, currently supports "BTC-USD-SPOT", "ETH-USD-SPOT", ...
    asset.ticker = ticker #               
    asset.set_start_date(startDateString) #       
    asset.interval = perSize                     
    asset.num_periods = nObs #                    Set number of reporting periods
    return(asset)

# ---

def callAndFormatHD(HDProvider = Coinalytix(), APIKeyArg = "yourKeyHere", assetParams = "parametersForAsset"):
    HD = HDProvider #                         Set api key
    HD.with_api_key(APIKeyArg) #               Fetch historical data
    asset_data = HD.fetch_hd(assetParams) #           Create Pandas data frame from result
    hddf = pd.DataFrame.from_dict(asset_data) # Adjust DatetimeIndex.
    hddf.set_index(pd.DatetimeIndex(hddf["StartDate"]*1000000000), inplace=True)
    return(hddf)

# --- 

def cleanAndPrintDataFrameSummary(thisDF = 'yourDataFrame', interpolateMethod = 'ffill', printSummary = True):
    """
    printSummary can be True, False, or 'justDim'
    """
    missCount = thisDF.isna().sum() #      get a sense of how many obs you're missing 
    missPerc = missCount / len(thisDF)
    thisDF = thisDF.fillna(method = interpolateMethod)
    if printSummary == True:
        print('')
        print("% of columns that needed to be interpolated")
        print(missPerc)                     
        #print("beginning of your data frame:")
        #print(thisDF.head(3))
        #print("end of your data frame:")      
        #print(thisDF.tail(3)) 
        print('')
        print("dimensions of your last DataFrame:")
        print(thisDF.shape) #
        print('')
    elif printSummary == "justDim":
        print(thisDF.shape) #
    return thisDF      
          
# ---   

def assetCallLoop(assetList = ['BTC-USD-SPOT', 'ETH-USD-SPOT'], exchangeList = ['BINANCE'],
                  perSize = "45m", startDateString = "2022-01-01 00:00:00", APIKey = 'YourAPIKeyHere', verbose = False):
    """
    """
    if startDateString == 'thisPeriod':
        startDateString = beginOfPeriod(thisTime = "now", lastTimeBloc = perSize)
    if len(exchangeList) == 1:
        exchangeList = np.repeat(exchangeList, len(assetList) )
    fin = {}        
    for i in range(len(assetList)):
        if verbose == True:
            print('calling ' + assetList[i])
        cw = formContempCallWindow(startDateString = startDateString, perSize = perSize) # call window 
        asset = define_asset_For_HDParams(exchange = exchangeList[i], ticker = assetList[i], startDateString = cw['startDateString'], perSize = cw['perSize'], nObs = cw['nObs'])
        print(cw['perSize'])
        hddf = callAndFormatHD(HDProvider = Coinalytix(), APIKeyArg = APIKey, assetParams = asset)
        if verbose == True and i == len(assetList)-1:
            printSummaryArg = True
        else:
            printSummaryArg = False
        hddf = cleanAndPrintDataFrameSummary(thisDF = hddf, printSummary = printSummaryArg)
        fin[ assetList[i] ] = hddf
        if verbose == True:
            if i == len(assetList)-1:
                print('...Call complete.')
    return fin


# ---

def whichTickers(baseURL = 'https://historicaldata.coinalytix.io/', apiKey = "", # = CA_API_KEY
                    verbose = False, Which = '', Which2 = '', WhichExch = ''): 
    '''
    call all coinalytix tickers & search by criteria
    Rohit's script turned into function
    Calls every ticker, then filters, so don't use in a loop
    '''
    tickerURL = baseURL + 'ohlc_tickers?api_key=' + apiKey
    r = requests.get(tickerURL)
    returnData = json.loads(r.text)
    df = pd.DataFrame(returnData['data']) # 
    
    if len(Which) > 0:
        df = df = df[ df['Ticker'].str.contains(pat = Which, regex = True) ]
    if len(Which2) > 0:
        df = df[ df['Ticker'].str.contains(pat = Which2, regex = True) ]
    if len(WhichExch) > 0:
        df = df[ df['Exchange'].str.contains(pat = WhichExch, regex = True) ]    
    if verbose == True: 
        print(df.head())
    return df

# ---

def fcs(timeBloc = '45m', nObsBack = 0, APIKey = 'CA_API_KEY', exchangeList = ['BINANCE', 'BINANCE'], assetList = ['BTC-USDT-SPOT', 'ETH-USDT-SPOT'],
        thisFunc = 'your_func', # you can add **vars here to pass to your func
        JRAPIKEY = 'JR_API_KEY', featureNames = ('BTCt', 'ETHt'), ippProc = ('last', 'last'), DV = ('BTC-USD', 'BTC-USD'), MBS = ('45', '45') ):
    """ fcs: Fetch, Calculate & Submit
    The first row of arguments structures your data call
    The second is the function you use to calculate your feature
    The third is (other than your API key) a set of tuples that you use to pass information to the JR API
    """
    startDateString = beginOfPeriod(thisTime = datetime.now(), timeBloc = timeBloc, nObsBack = nObsBack)
    XDict = assetCallLoop(exchangeList = exchangeList, assetList = assetList, startDateString = startDateString, perSize = timeBloc, APIKey = APIKey, verbose = False)
    i = 0
    for k, v in XDict.items():
        x = thisFunc(v)
        JR = JudgeResearch()
        JR.with_api_key(JRAPIKEY)
        submissionPars = JRParams()
        submissionPars.dv = DV[i]
        submissionPars.mbs= MBS[i]   
        submissionPars.feature_name = featureNames[i]
        submissionPars.ipp = ippProc[i]
        features = JR.craft_features(default_params = submissionPars, df = x)
        payload = JR.format_payload(features)
        submit = JR.submit_feature(payload)
        print(submit)
        i = i + 1 

# ---

def live_fcs(firstdelay, seconddelay, **A):
        """ live_fcs:  
        calculate and submit feature data 3 times, throttled by percentage
        of time remaining until end of window
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
        window = timedelta(minutes=int(A['MBS'][1])) #                  4. create timedelta object based on MBS parameter
        while now > deadline: #                                        5. iterate through block times until current time is passed, set deadline
            deadline = deadline + window
        print("Next Block End: " + deadline.strftime("%Y-%m-%dT%H:%M:%SZ"))
        remaining = deadline - now #                                   6. calculate time remaining until current window closes
        delay1 = int(remaining.total_seconds() * firstdelay) #         7. calculate delays (seconds)
        delay2 = int(remaining.total_seconds() * seconddelay) - delay1
        
        ##### Sec. 2:  use fcs (fetch, calculate & submit) to send in features three times #####
        s1 = fcs(timeBloc = A['timeBloc'], nObsBack = A['nObsBack'], APIKey = A['APIKey'], exchangeList = A['exchangeList'], assetList = A['assetList'],
        thisFunc = A['thisFunc'], JRAPIKEY = A['JRAPIKEY'], featureNames = A['featureNames'], ippProc = A['ippProc'], DV = A['DV'], MBS = A['MBS']) # 1. immediately
        print("First submission at " + datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"))
        print(s1)
        
        time.sleep(delay1) #                                           2. sleep, & run fcs after delay 1
        s2 = fcs(timeBloc = A['timeBloc'], nObsBack = A['nObsBack'], APIKey = A['APIKey'], exchangeList = A['exchangeList'], assetList = A['assetList'],
        thisFunc = A['thisFunc'], JRAPIKEY = A['JRAPIKEY'], featureNames = A['featureNames'], ippProc = A['ippProc'], DV = A['DV'], MBS = A['MBS'])
        print("Second submission at " + datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"))
        print(s2)
        
        time.sleep(delay2) #                                           3. and a final time.
        s3 = fcs(timeBloc = A['timeBloc'], nObsBack = A['nObsBack'], APIKey = A['APIKey'], exchangeList = A['exchangeList'], assetList = A['assetList'],
        thisFunc = A['thisFunc'], JRAPIKEY = A['JRAPIKEY'], featureNames = A['featureNames'], ippProc = A['ippProc'], DV = A['DV'], MBS = A['MBS'])
        print("Final submission at " + datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"))        
        print(s3)


### ----- Section 3. Matrix Manipulation, Data Wrangling, etc.  ------ ###

def lagMat(x, lag = 2, mVM = "0", inct = False, shortColN = 0): 
    """
    mVM:  missing value method - '0' (replace w/ zeros); 'drop' = remove rows
    improve to take a vector of lags, not range
    inct: include t, the vector to be lagged
    shortColN: shorten column name.  0 is no shortening, 1 is just the first letter.
    """
    if type(x) is pd.DataFrame:
        newDict = {}
        for colName in x: #                            handle column names
            if inct == True:
                if shortColN == 0: 
                    newDict[colName] = x[colName]
                else: 
                    newDict[ colName[:shortColN] ] = x[colName]
            for l in range(1,lag+1): #                 create lagged Series
                if shortColN == 0: 
                    newDict['%sL%d' %(colName,l)] = x[colName].shift(l)
                else:
                    newDict['%sL%d' %(colName[:shortColN],l)] = x[colName].shift(l)
        res = pd.DataFrame(newDict,index = x.index)
    elif type(x) is pd.Series:
        theRange = range(lag + 1)
        res = pd.concat([x.shift(i) for i in theRange],axis = 1)
        res.columns = ['L%d' %i for i in theRange]
    else:
        print('Only works for DataFrame or Series')
        return None
    if not inct and type(x) is not pd.DataFrame:
        res = res.drop(columns=['L0'])
    if mVM == "drop":
        return res.dropna()
    elif mVM == "0":
        return res.fillna(0)
    else:
        return res 
# ---



### ----- Section 4. Stat Analysis & Machine Learning ------ ###

# ---

def perDif(x):
    """
    takes an ohlc object & returns % difference
    """
    pd = (x['Open'] - x['Close']) / x['Open']
    return pd

# --- moving average to % change - % deviated from MA

def mAP1(x, mACol = 7, cbind = True, cN = 'mAP'): # this shit aint done, right?
    """ moving average to standardized 
    Takes in OHLCV + MACD dataframe, returns tscore
    mACol:  moving average column to use as the mean in the standardization 
    """
    maDifPerc = (x['Close'] - x.iloc[:,7]) / x['Open'] - 1
    if cbind:
        maDifPerc = cb(x, maDifPerc)    
        maDifPerc.columns = [*maDifPerc.columns[:-1], cN]
    return maDifPerc

# ---

def ols_olsRan(modObj = 'modToRun', retSum = True): 
    '''
    Just saves you lines of code if you're specifying & running lots of models
    '''
    dict2Ret = {'obj':'', 'run':'','r':'','f':'', 'sum':''}
    dict2Ret['obj'] = modObj
    dict2Ret['run'] = modObj.fit()
    dict2Ret['r'] = dict2Ret['run'].resid
    dict2Ret['f'] = dict2Ret['run'].fittedvalues
    if retSum == True:
        dict2Ret['sum'] = dict2Ret['run'].summary() # else drop dict@ret['sum']
    return(dict2Ret)

def manyOLS(funcDict, datMat, retSummary = True, concatFits = True, concatRes = True  ):
    fin = {}
    if concatFits == True:
        fM = {} #         forecast matrix
    if concatRes == True:
        rM = {} #         residuals matrix
    for k, v in funcDict.items():
        fin[k] = ols_olsRan( smf.ols(formula = v, data = datMat), retSum = retSummary ) 
        if concatFits == True:
            fM[k] = fin[k]['f'] 
        if concatRes == True:
            rM[k] = fin[k]['r'] 
    if concatFits == True:
        fM = pd.DataFrame(fM)
        fM = fM.add_suffix('f')
        fin['fM'] = fM
    if concatFits == True:
        rM = pd.DataFrame(rM)
        rM = rM.add_suffix('r')
        fin['rM'] = rM
    if concatFits == True and concatRes == True:
        fin['rfM'] = pd.concat([fM, rM], axis = 1)
    return(fin)
    