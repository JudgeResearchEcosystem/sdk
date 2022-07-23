from datetime import datetime

import pandas as pd
import requests


class HDParams:
    """
    Historical data parameters
    """

    def __init__(
        self,
        exchange="BINANCE",
        ticker="BTC-USD-SPOT",
        start_date="2022-01-01 08:00:00",
        interval="1d",
        num_periods="365",
    ):

        self.exchange = exchange
        self.ticker = ticker
        self.start_date = datetime.fromisoformat(start_date)
        self.interval = interval
        self.num_periods = num_periods

    def set_start_date(self, datestr):
        """format the start date string for Coinalytix"""
        self.start_date = datetime.fromisoformat(datestr)


class Coinalytix:
    """
    Conduit to Coinalytix API
    """

    def __init__(self):
        self.url_root = 'https://historicaldata.coinalytix.io/ohlc?' # old url:  "https://timeseries.coinalytix.io/getohlcdata?" # 
        self.api_key = ""

    def with_api_key(self, key):
        """set API key for Coinalytix authentication"""
        self.api_key = key # 

    def fetch_hd(self, asset):
        """fetch historical data dictionary"""
        _url = (
            self.url_root
            + "api_key="
            + self.api_key
            + "&exchange="
            + asset.exchange
            + "&ticker="
            + asset.ticker
            + "&startDate="
            + str(int(asset.start_date.timestamp() )) 
            + "&interval="
            + asset.interval
            + "&numPeriods="
            + str(asset.num_periods)
        )
        r = requests.get(_url)
        if r.status_code == 200:
            _json = r.json()
            data = _json["data"]

            # convert dict to pandas df
            hddf = pd.DataFrame.from_dict(data)

            # set DateTimeIndex
            hddf.set_index(
                pd.DatetimeIndex(hddf["StartDate"] * 1000000000), inplace=True
            )

            return hddf

        else:
            print("Error fetching data")
            return r.text
