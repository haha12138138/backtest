from typing import Optional

import pandas as pd


def merge_data_by_date(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    merged_data = dfs[0]
    for df in dfs[1:]:
        merged_data = pd.merge(merged_data, df, on='date', how='outer')
    merged_data.sort_index(inplace=True)
    return merged_data


# esitmate the trading date using business dates.
def get_business_dates(start: Optional[str] = None, end: Optional[str] = None, count: Optional[int] = None,
                       freq: str = 'B') -> tuple[str, str]:
    if len([arg for arg in (start, end, count) if arg is not None]) != 2:
        raise ValueError("Exactly 2 out of 3 arguments must be specified.")
    else:

        if freq == "B":
            if start is None:
                date_range = pd.date_range(end=end, periods=count, freq=freq, inclusive="both")
            elif end is None:
                date_range = pd.date_range(start=start, periods=count, freq=freq, inclusive="both")
            else:
                date_range = pd.date_range(start=start, end=end, freq=freq, inclusive="both")
        elif False:
            if start is None:
                date_range = pd.date_range(end=end, periods=count+1, freq=freq, inclusive="left")
            elif end is None:
                date_range = pd.date_range(start=start, periods=count+1, freq=freq, inclusive="right")
            else:
                date_range = pd.date_range(start=start, end=end, freq=freq, inclusive="both")
        else:
            raise ValueError("freq are supported only for daily (B)")

    return str(date_range[0].strftime('%Y-%m-%d')), str(date_range[-1].strftime('%Y-%m-%d'))


def get_filtered_date_data(df=None
                           , start_date: Optional[str] = None
                           , end_date: Optional[str] = None
                           , count: Optional[int] = None
                           , freq: str = 'B'):
    """
        this function calculates data_start_date and data_end_date of a data. this function requires 2 of 3 args among "data_start_date"
        "data_end_date" "count"
        :param df:
        :param start_date:
        :param end_date:
        :param count:
        :param freq:
        :return: a truncate the dataframe if df is not none. a tuple (data_start_date, data_end_date) will be returned.
        """
    start_date, end_date = get_business_dates(start_date, end_date, count, freq)
    if df is not None:
        return df.loc[(df.index >= start_date) & (df.index <= end_date)]
    else:
        return start_date, end_date
