import warnings
from typing import Optional

import pandas as pd
from common_util import merge_data_by_date, get_business_dates, get_filtered_date_data
from fmp import *


class stock_database:
    def __connect_local_database(self, database_credential):
        pass

    def __load_local_files(self, local_file_list):
        if local_file_list is None:
            return dict()
        else:
            dfs = dict()
            for i in local_file_list:
                df = pd.read_csv(i[1])
                df['date'] = pd.to_datetime(df['date'])
                dfs[i[0]] = df
            return dfs

    def __init__(self, local_file_list=None, local_database_credential=None, network_apikey=None):
        '''
            local_file_list has a type of [(string,string)] first string is the ticker and the second one is the file location
        '''
        # There are 3 data sources. we will use localfiles as first primary choice. then the local database, then the network API
        if local_file_list is None and local_database_credential is None and network_apikey is None:
            raise ValueError(
                "At least one of the arguments: local_file, local_database_credential, network_apikey must not be None"
            )
        self.local_files = self.__load_local_files(local_file_list)
        self.local_database = self.__connect_local_database(local_database_credential)
        self.network_apikey = network_apikey

    # def __get_filtered_date_data(self, df=None, data_start_date: Optional[str] = None, data_end_date: Optional[str] = None,
    #                              count: Optional[int] = None):
    #     """
    #     this function calculates data_start_date and data_end_date of a data. this function requires 2 of 3 args among "data_start_date"
    #     "data_end_date" "count"
    #     :param df: this arg is only needed when data source is local file
    #     :param data_start_date:
    #     :param data_end_date:
    #     :param count:
    #     :return: a truncate the dataframe if df is not none. a tuple (data_start_date, data_end_date) will be returned.
    #     """
    #     data_start_date, data_end_date = get_business_dates(data_start_date, data_end_date, count)
    #     if df is not None:
    #         return df.loc[(df.index >= data_start_date) & (df.index <= data_end_date)]
    #     else:
    #         return data_start_date, data_end_date

    def get_stock_price(self, tickers: list[str]
                        , fields: list
                        , start_date: Optional[str] = None
                        , end_date: Optional[str] = None
                        , count: Optional[int] = None
                        , period: str = '1d'
                        , adjustment: str = 'back'
                        ,data_source: str = "") -> pd.DataFrame:

        # input arg validation
        if period != '1d' or adjustment != 'back' or data_source != "":
            raise NotImplementedError(
                "[NOT IMPL] period only support 1d, adjustment only support back data_source should be empty, but get "
                + "period = " + str(period) + " " + "adjustment = " + str(adjustment) + " " + "data_source = " + str(
                    data_source))

        # Check which database to use. only check if ticker exists
        # TODO select source based on whether have latest data and whether a ticker is included in the DB
        ticker_database_map = {"local_files": [], "local_database": [], "network_api": []}
        for ticker in tickers:
            if ticker in self.local_files.keys():
                ticker_database_map["local_files"].append(ticker)
            elif False and (self.local_database.exists_ticker(ticker)):
                #TODO local_database need to include exists_ticker method. masking this now
                raise NotImplementedError(
                    "[NOT IMPL] only support local filelist. please check the ticker and the datafiles")
            elif fmp_check_ticker_exists(ticker,apikey=self.network_apikey,only_active_trading=False):
                #temporarily bypass a api bug. if a company is not actively traded, the API will not return any price data.
                #only_active_trading should be set to FALSE to ensure no future bias.
                ticker_database_map["network_api"].append(ticker)
            else:
                warnings.warn(f"This ticker: {ticker} cannot be found!")

        # We start_date to read data from different sources and then combine them together.
        dfs = []
        # first get the needed ticker and the field from the database.
        # return will be a list of field data of tickers
        for ticker in ticker_database_map['local_files']:
            df = self.local_files[ticker]
            df.columns =df.columns.str.lower()
            df = (df[fields + ['date']])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = get_filtered_date_data(df, start_date=start_date, end_date=end_date, count=count)
            df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
            dfs.append(df)

        for ticker in ticker_database_map['local_database']:
            pass

        for ticker in ticker_database_map['network_api']:
            s, e = get_filtered_date_data(start_date=start_date, end_date=end_date, count=count)
            print(ticker)
            df = fmp_get_price_of_a_stock(ticker, fields, apikey=self.network_apikey, start=s, end=e)
            dfs.append(df)

        # align the database by date. the number of rows depends on the stock with longest history
        merged_data = merge_data_by_date(dfs)
        # now we apply the date filter.
        return merged_data

    def get_fundamentals(self
                         , tickers: [str]
                         , fields: list[Union[Income_statement_entry, Balance_statement_entry, Cashflow_statement_entry, Financial_ratio_entry]]
                         # , apikey: str
                         , start_date: Optional[str] = None
                         , end_date: Optional[str] = None
                         , count: Optional[int] = None
                         , period: Financial_statement_period = Financial_statement_period.annual
                         ) -> pd.DataFrame:
        dfs = dict()
        for ticker in tickers:
            dfs[ticker] = (fmp_get_fundamentals_of_a_stock(ticker
                                                        , fields
                                                        , self.network_apikey
                                                        , start_date
                                                        , end_date
                                                        , count
                                                        , period))
        return dfs #merge_data_by_date(dfs)
