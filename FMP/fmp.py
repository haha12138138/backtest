import asyncio
import datetime
import logging
import warnings
from enum import Enum
from typing import Optional, Union, Literal, List, Dict

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta

from common_util import merge_data_by_date
from fmp_api_access_base import __api_access, BASE_URL_v3, BASE_URL_v4, __api_access_async
from fmp_datatypes import Mkt_index, Price_entry, Income_statement_entry, Financial_ratio_entry, \
    Balance_statement_entry, Cashflow_statement_entry, Financial_statement_period, Forcast_entry


# def fmp_check_ticker_exists(ticker: str, apikey:str):
#     url = f"{BASE_URL_v3}/search-ticker?query={ticker}&limit=1&apikey={apikey}"
#     res = __api_access(url)
#     return len(res) != 0

def fmp_check_ticker_exists(ticker: str, apikey: str, only_active_trading=False):
    url = f"{BASE_URL_v3}/profile/{ticker}?apikey={apikey}"
    res = __api_access(url)
    if len(res) == 0:
        return False
    elif (res[0]["isActivelyTrading"] == False) and (only_active_trading == True):
        warnings.warn(f"This ticker: {ticker} is not actively traded. only_active_trading flag is set to True")
        return False
    else:
        return True


def fmp_get_industry(apikey: str):
    url = f"{BASE_URL_v4}/standard_industrial_classification/all?apikey={apikey}"
    res = __api_access(url)
    df = pd.DataFrame(res)
    df = df.set_index("symbol")
    return df


def get_mkt_constituents(index: Mkt_index, apikey) -> tuple[str, list]:
    url = f"{BASE_URL_v3}/{index.value}?apikey={apikey}"
    res = __api_access(url)
    df = pd.DataFrame(res)
    # we only need, symbol, sector and subsector
    tickers = df["symbol"].values.tolist()
    return str(datetime.date.today()), tickers


def construct_mkt_historical_constituents(index: Mkt_index, constituents, apikey):
    url_hist = f"{BASE_URL_v3}/historical/{index.value}?apikey={apikey}"
    res = __api_access(url_hist)
    last_change_date = constituents[0]
    all_historical_constituents = []
    next_constituent_values = dict()
    for change in res:
        change_date = change["date"]
        change_symbol = change["symbol"]
        if change_date != last_change_date:
            if len(all_historical_constituents) == 0:
                all_historical_constituents = [constituents]
                next_constituent_values = {i: None for i in constituents[1]}
                last_change_date = change_date
            else:
                all_historical_constituents.append(((datetime.datetime.strptime(last_change_date,
                                                                                "%Y-%m-%d") - datetime.timedelta(
                    days=1)).strftime("%Y-%m-%d")
                                                    , list(next_constituent_values.keys())))
                last_change_date = change_date
        if change["removedTicker"] == "":
            # added into the index on this date. we need to remove it from the constituents
            del next_constituent_values[change_symbol]
        else:
            next_constituent_values[change_symbol] = None
    return all_historical_constituents


def fmp_get_etf_holdings(ticker: str, apikey: str, start_date: Optional[str] = None, end_date: Optional[str] = None,
                         highest_N=10):
    url = f"{BASE_URL_v4}etf-holdings/portfolio-date?symbol={ticker}&apikey={apikey}"
    res = __api_access(url)
    list_of_dates = [pd.to_datetime(i["date"], format="%Y-%m-%d") for i in res]
    start_date = pd.to_datetime(start_date, format="%Y-%m-%d")
    end_date = pd.to_datetime(end_date, format="%Y-%m-%d")
    list_of_date_in_date_range = list(filter(lambda x: (x >= start_date) and (x <= end_date), list_of_dates))
    df_holding = pd.DataFrame(columns=["date", "holdings"])
    verified_list = []
    failed_list =[]
    if list_of_date_in_date_range != []:
        for date in list_of_date_in_date_range:
            url = f"{BASE_URL_v4}etf-holdings?date={date.strftime('%Y-%m-%d')}&symbol={ticker}&apikey={apikey}"
            res = __api_access(url)
            df_res = pd.DataFrame(res)[["symbol", "pctVal"]].sort_values(by="pctVal", ascending=False).iloc[0:highest_N]
            holding_list = df_res["symbol"].values.tolist()
            holding_list = list(filter(lambda x: x is not None, holding_list))
            verified_set = set(verified_list)
            failed_set = set(failed_list)
            holding_list_temp = []

            for holding in holding_list:
                if holding in verified_set:
                    holding_list_temp.append(holding)
                elif holding not in failed_set:
                    if fmp_check_ticker_exists(holding, apikey):
                        verified_set.add(holding)
                        holding_list_temp.append(holding)
                    else:
                        failed_set.add(holding)

            # Update the lists with the sets for future iterations
            verified_list = list(verified_set)
            failed_list = list(failed_set)


            df_holding.loc[len(df_holding)] = {"date": date, "holdings": holding_list_temp}
    if start_date < min(list_of_date_in_date_range):
        df_oldest_holdings = df_holding.loc[df_holding["date"] == min(list_of_date_in_date_range)]
        oldest_holdings = df_oldest_holdings["holdings"].tolist()[0]
        holding_list_temp = []
        for holding in oldest_holdings:
            if fmp_check_ticker_exists(holding, apikey) == True:
                holding_list_temp.append(holding)

        df_holding.loc[len(df_holding)] = {"date": start_date, "holdings": holding_list_temp}
    return df_holding.set_index("date").sort_index(ascending=True)


def fmp_get_mktcap_of_a_stock(ticker: str
                              , apikey: str
                              , start: Optional[str] = None
                              , end: Optional[str] = None):
    """
    This is a function that gets daily mktcap of a stock.
    mktcap is calculated by multiplying the current share price by the number of outstanding shares.
    :param ticker:
    :param apikey:
    :param start: data start_date date. (if both start_date and end_date are None, fetch the latest mktcap data)
    :param end: data end_date date. (if both start_date and end_date are None, fetch the latest mktcap data)
    :return: dataframe that use ticker as column name and date as index. date is formatted as "YYYY-mm-dd". date is sorted
    in ascending order (oldest date first)
    """
    if (start is None) and (end is None):
        # get only the latest data.
        url = f"{BASE_URL_v3}/market-capitalization/{ticker}?apikey={apikey}"
        res = __api_access(url)
        df = pd.DataFrame(pd.DataFrame(res).set_index("date")["marketCap"]).rename({"marketCap": ticker},
                                                                                   axis="columns")
        return df
    elif (start is not None) and (end is not None):
        df_list = []
        url_base = f"{BASE_URL_v3}/historical-market-capitalization/{ticker}?"
        start_date = datetime.datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(end, "%Y-%m-%d")
        temp_start_date = end_date - relativedelta(years=5)
        while temp_start_date > start_date:
            temp_start = temp_start_date.strftime("%Y-%m-%d")
            end = end_date.strftime("%Y-%m-%d")
            url = f"{url_base}from={temp_start}&to={end}&apikey={apikey}"
            res = __api_access(url)
            if len(res) != 0:
                df_list.append(pd.DataFrame(res).set_index("date")["marketCap"])
            end_date = temp_start_date - relativedelta(days=1)
            temp_start_date = end_date - relativedelta(years=5)
        # temp_start_date <= data_start_date
        end = end_date.strftime("%Y-%m-%d")
        url = f"{url_base}from={start}&to={end}&apikey={apikey}"
        res = __api_access(url)
        if len(res) != 0:
            df_list.append(pd.DataFrame(res).set_index("date")["marketCap"])

        if len(df_list) == 0:
            df_temp = pd.DataFrame(columns=[ticker])
            df_temp.index.name = "date"
            df_list.append(df_temp)
        return pd.DataFrame(pd.concat(df_list)).sort_index(ascending=True).rename({"marketCap": ticker}, axis="columns")


def fmp_get_price_of_a_stock(ticker: str
                             , fields: list[Price_entry]
                             , apikey: str
                             , start: Optional[str] = None
                             , end: Optional[str] = None
                             ) -> pd.DataFrame:
    """
    :param fields:
    :param ticker: a string that contains the ticker of a stock
    :param start: start_date date of the data (if that day is holiday or weekend, it will be the first day after)
    :param end: end_date date of the data (if that day is holiday or weekend, it will be the first day before)
    :return: a multi-index dataframe.
            the 1st level col is ticker
            , the 2nd level contains fields selected in arg field (eg: open/close/high/low/volumn)
    """
    url = f"{BASE_URL_v3}/historical-price-full/{ticker}?apikey={apikey}"
    if start is not None:
        url += f"&from={start}"
    if end is not None:
        url += f"&to={end}"
    res = __api_access(url)
    field_names = list(map(lambda x: x.name, fields))
    if len(res) != 0:
        df = pd.DataFrame(res["historical"])
        df['date'] = pd.to_datetime(df['date'])
        df.rename(columns={x.value: x.name for x in fields}, inplace=True)

        df = df[field_names + ['date']].set_index('date')
        df.columns = pd.MultiIndex.from_product([[res['symbol']], df.columns])
    else:
        df = pd.DataFrame(data=None, columns=field_names)
        df.index.name = 'date'
        df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
    return df


def __fmp_get_statement(ticker: str
                        , type: str
                        , fields: Union[
            list[Income_statement_entry], list[Balance_statement_entry], list[Cashflow_statement_entry], list[
                Financial_ratio_entry]]
                        , apikey: str
                        , start: Optional[str] = None
                        , end: Optional[str] = None
                        , count: Optional[int] = None
                        , period: Financial_statement_period = Financial_statement_period.annual
                        ) -> pd.DataFrame:
    """
    This is a helper function of fmp_get_fundamentals_of_a_stock. It can only get 1 type of statement.
    """
    print(fields)
    # check the type of field.
    if type == "I":
        url = f"{BASE_URL_v3}/income-statement/{ticker}?period={period.value}&apikey={apikey}"
    elif type == "B":
        url = f"{BASE_URL_v3}/balance-sheet-statement/{ticker}?period={period.value}&apikey={apikey}"
    elif type == "C":
        url = f"{BASE_URL_v3}/cash-flow-statement/{ticker}?period={period.value}&apikey={apikey}"
    elif type == "R":
        url = f"{BASE_URL_v3}/ratios/{ticker}?period={period.value}&apikey={apikey}"
    else:
        raise ValueError(
            "field can only be chosen among income_statement_entry balance_statement_entry cashflow_statement_entry")
    field_name = list(map(lambda x: x.value, fields))
    res = __api_access(url)
    if len(res) != 0:
        df = pd.DataFrame(res)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df.sort_index(ascending=False, inplace=True)
        if start is None:
            df = df[df.index <= end][:count]
        elif end is None:
            df = df[df.index >= start][-(count):]
        else:
            df = df[(df.index >= start) & (df.index <= end)]
        # df = get_filtered_date_data(df
        #                             , data_start_date=start_date, data_end_date=end_date, count=count
        #                             , freq="3M" if period == Financial_statement_period.quarter else "12M")
        # fetch necessary fields
        df = df[field_name]
        # convert the name to enum name
        df.rename(columns={x.value: x.name for x in fields}, inplace=True)
        df.columns = pd.MultiIndex.from_product([[res[0]['symbol']], df.columns])
    else:
        df = pd.DataFrame(data=None, columns=field_name)
        df.index.name = 'date'
        df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
    return df


def fmp_get_fundamentals_of_a_stock(ticker: str
                                    , fields: list[
            Union[Income_statement_entry, Balance_statement_entry, Cashflow_statement_entry, Financial_ratio_entry]]
                                    , apikey: str
                                    , start: Optional[str] = None
                                    , end: Optional[str] = None
                                    , count: Optional[int] = None
                                    , period: Financial_statement_period = Financial_statement_period.annual
                                    ) -> pd.DataFrame:
    """
    This function is to retrieve financial information. this includes, balance sheet, cashflow sheet, income sheet
    along with other useful financial ratios.

    :param ticker: stock name list
    :param fields: a list of financial information
    :param apikey:
    :param start: start_date date
    :param end: end_date date
    :param count: the length of data measured by "period" arg. So it is possible that 2 results are given even though count == 1
    :param period: whether you are retrieve quarterly result or annual result
    :return: a multi-level index dataframe.
             index is the date (timestamp, datetime ...)
             1st level of column is ticker, 2nd level constains the name of financial information you searched for

    """
    # group field by their type
    field_type: dict = {"I": [], "B": [], "C": [], "R": []}
    for field in fields:
        if field.name in Income_statement_entry.__members__.keys():
            field_type['I'].append(field)
        elif field.name in Balance_statement_entry.__members__.keys():
            field_type['B'].append(field)
        elif field.name in Cashflow_statement_entry.__members__.keys():
            field_type['C'].append(field)
        elif field.name in Financial_ratio_entry.__members__.keys():
            field_type['R'].append(field)
        else:
            raise ValueError("unrecognized field type: " + str(type(field)))
    dfs = []
    for k, v in field_type.items():
        if v == []:
            continue
        dfs.append(__fmp_get_statement(ticker, type=k, fields=v, apikey=apikey
                                       , start=start, end=end, count=count
                                       , period=period))
    return merge_data_by_date(dfs)

def fmp_get_forecast(ticker: str
                     , fields: list[Forcast_entry]
                     , apikey:str
                     , start: Optional[str] = None
                     , count: Optional[int] = None
                     , period: Financial_statement_period = Financial_statement_period.annual) -> pd.DataFrame:
    url = f"{BASE_URL_v3}/analyst-estimates/{ticker}?period={period.value}&apikey={apikey}"
    field_name = list(map(lambda x: x.value, fields))
    res = __api_access(url)
    if len(res) != 0:
        df = pd.DataFrame(res)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df.sort_index(ascending=False, inplace=True)
        df = df[df.index >= start][-(count):]
        df = df[field_name]
        df.rename(columns={x.value: x.name for x in fields}, inplace=True)
        df.columns = pd.MultiIndex.from_product([[res[0]['symbol']], df.columns])
    else:
        df = pd.DataFrame(data=None, columns=field_name)
        df.index.name = 'date'
        df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
    return df


def fmp_get_sec_fillings(
    ticker: Union[str, List[str]],
    filling_type: Literal["10-k", "10-q"],
    apikey: str,
    prev_n: Union[int, Literal["all"]] = "all"
) -> Dict[str, Union[List, Dict]]:
    """
    This function returns a bunch of SEC filings with some metadata: symbol, acceptedDate, type.
    It uses the __api_access_async function internally to make asynchronous API calls.

    :param ticker: A single ticker or a list of tickers.
    :param filling_type: The type of SEC filing ("10-k" or "10-q").
    :param apikey: The API key for authentication.
    :param prev_n: A number or "all". Default is "all".
    :return: A dictionary where keys are tickers and values are the API responses.
    """
    if not isinstance(ticker, list):
        ticker = [ticker]

    # Prepare the URLs for each ticker
    urls = [
        f"{BASE_URL_v3}/sec_filings/{t}?type={filling_type}&page=0&apikey={apikey}"
        for t in ticker
    ]

    # Define an async function to handle the API calls
    async def _fetch_all():
        tasks = [__api_access_async(url) for url in urls]
        results = await asyncio.gather(*tasks)
        if prev_n == "all":
            return {t: result for t, result in zip(ticker, results)}
        else:
            return {t: result[0:prev_n] for t, result in zip(ticker, results)}

    # Run the async function in an event loop and wait for it to complete
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(_fetch_all())