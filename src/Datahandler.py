from typing import Type

import numpy as np

from FMP.fmp import *
from FMP.main import stock_database
from Strategy import Strategy
from Universe import Stock_Universe
from fmp_api_access_base import API_KEY
from fmp_datatypes import Datapoint_period


class DataHandler:
    def __init__(self, universe: Stock_Universe, data_start_date: str, data_end_date: str
                 , strategy):
        self.universe = universe
        self.start_date = data_start_date
        self.end_date = data_end_date
        self.strategy = strategy
        # self.data = self.loaddata(date_requirements)
        self.current_time_index = 0
        # 使用价格数据的日期索引作为时间
        self.data = None
        self.aq_mapper = dict()

    def __parse_strategy_dependencies(self, strategy: Type[Strategy]):

        daily_entries = []
        quarter_entries = []
        max_num_needed_for_quarterly_data = 1
        annual_entries = []
        max_num_needed_for_annually_data = 1
        max_num_needed_for_daily_data = 1
        for dependency in strategy.inherent_dependencies:
            if isinstance(dependency.dependence_name, Price_entry):
                daily_entries.append(dependency.dependence_name)
                max_num_needed_for_daily_data = max(max_num_needed_for_daily_data,
                                                    dependency.num_of_data_needed)
            else:
                if dependency.period.value == Datapoint_period.annual.value:
                    annual_entries.append(dependency.dependence_name)
                    max_num_needed_for_annually_data = max(max_num_needed_for_annually_data,
                                                           dependency.num_of_data_needed)
                elif dependency.period.value == Datapoint_period.quarter.value:
                    quarter_entries.append(dependency.dependence_name)
                    max_num_needed_for_quarterly_data = max(max_num_needed_for_quarterly_data,
                                                            dependency.num_of_data_needed)
                else:
                    ValueError(f"{dependency} is not correct")
        return list(set(daily_entries)), list(set(quarter_entries)), list(
            set(annual_entries)), max_num_needed_for_daily_data, max_num_needed_for_quarterly_data, max_num_needed_for_annually_data

    def __calc_start_date_for_fundamental_data(self, start_time, max_num_needed_for_quarterly_data,
                                               max_num_needed_for_annually_data):
        extra_days_needed = max(max_num_needed_for_quarterly_data * 90, max_num_needed_for_annually_data * 365)
        return (datetime.datetime.strptime(start_time, "%Y-%m-%d") - datetime.timedelta(extra_days_needed)).strftime(
            "%Y-%m-%d")

    def __calc_start_date_for_price_data(self, start_time, max_num):
        extra_days_needed = max_num + 5  # hope this is large enough to cover weekend and holidays
        return (datetime.datetime.strptime(start_time, "%Y-%m-%d") - datetime.timedelta(extra_days_needed)).strftime(
            "%Y-%m-%d")
    def loaddata(self):

        # 这里调用外部API获取数据，返回一个dict=None
        db = stock_database(network_apikey=API_KEY)
        # price_entries = list(filter(lambda x: isinstance(x, Price_entry), self.data_requirements))
        # fund_entries = list(filter(lambda x: isinstance(x, (Balance_statement_entry
        #                                                     , Cashflow_statement_entry
        #                                                     , Income_statement_entry
        #                                                     , Financial_ratio_entry)), self.data_requirements))
        price_entries, quarter_entries, annual_entries, max_num_needed_for_daily_data, max_num_needed_for_quarterly_data, max_num_needed_for_annually_data = self.__parse_strategy_dependencies(
            self.strategy)
        start_date_for_daily_data = self.__calc_start_date_for_price_data(self.start_date,
                                                                          max_num_needed_for_daily_data)
        self.universe.load_data(start_date_for_daily_data, self.end_date)
        df_price = db.get_stock_price(self.universe.get_total_universe()
                                          , fields=price_entries
                                          , end_date=self.end_date
                                          , start_date=start_date_for_daily_data)

        start_date_for_fundamentals = self.__calc_start_date_for_fundamental_data(self.start_date,
                                                                                  max_num_needed_for_quarterly_data,
                                                                                  max_num_needed_for_annually_data)
        if quarter_entries != []:
            dict_fund_q = db.get_fundamentals(self.universe.get_total_universe()
                                              , fields=quarter_entries
                                              , end_date=self.end_date
                                              , start_date=start_date_for_fundamentals
                                              , period=Financial_statement_period.quarter)
        else:
            dict_fund_q = dict()

        if annual_entries != []:
            dict_fund_a = db.get_fundamentals(self.universe.get_total_universe()
                                              , fields=annual_entries
                                              , end_date=self.end_date
                                              , start_date=start_date_for_fundamentals
                                              , period=Financial_statement_period.annual)
        else:
            dict_fund_a = dict()

        for k in quarter_entries:
            self.aq_mapper[k.name] = "fundamental_q"
        for k in annual_entries:
            self.aq_mapper[k.name] = "fundamental_a"

        # 为了示例，假设data_dict已经被加载并符合要求
        data_dict = {
            'price': df_price,  # 价格数据
            'fundamental_q': dict_fund_q,  # 财务数据
            'fundamental_a': dict_fund_a,
            'macro': pd.DataFrame(),  # 宏观数据
        }
        # 数据预处理，处理NaN值，只进行前向填充
        # for key in data_dict:
        #     data_dict[key] = data_dict[key].ffill()

        self.data = data_dict
        self.time_index = self.data['price'].index
        self.current_date = datetime.datetime.strptime(self.start_date,
                                                       "%Y-%m-%d")  # self.time_index[self.current_time_index]
        self.current_time_index = np.where(self.time_index <= self.current_date)[0][-1]
        self.current_date = self.time_index[self.current_time_index]
    def reset(self):
        self.current_time_index = 0
        self.current_date = datetime.datetime.strptime(self.start_date,
                                                       "%Y-%m-%d")  #self.time_index[self.current_time_index]

    def get_dynamic_universe(self, name="default_group"):
        df = self.universe.get_universe(name)
        df = df[df.index < self.current_date]
        return df["holdings"].values.tolist()[-1]

    def get_price(self, metric: Price_entry, prev=1, allow_today_data=False):
        df = self.data["price"]

        # 确保不获取未来数据，只获取当前日期之前的数据
        if allow_today_data == False:
            df = df[df.index < self.current_date]  # latest would be yesterday's data
        else:
            df = df[df.index <= self.current_date]  # the last one would be today's data, for backtester ONLY

        # 检查数据长度是否足够
        if len(df) < prev:
            return None  # 数据不足，返回None

        # 获取前N个数据
        result = df.iloc[-prev:]

        # 选择指标列，假设列是MultiIndex (ticker, metric)
        if metric.name in result.columns.get_level_values(1):
            data = result.xs(metric.name, level=1, axis=1)
            return data
        else:
            return None  # 未找到对应的指标

    def get_fundamentals(self, metric: Union[Income_statement_entry, Balance_statement_entry, Cashflow_statement_entry
    , Financial_ratio_entry], prev=1):
        dict_fund = self.data[self.aq_mapper[metric.name]]
        dfs = []
        for ticker, df in dict_fund.items():
            if len(df) == 0:
                continue
            df = df[df.index < self.current_date]
            if len(df) < prev:
                continue
            dfs.append(df.iloc[-prev:].reset_index(drop=True))

        if dfs == []:
            return None

        result = pd.concat(dfs, axis=1)

        if metric.name in result.columns.get_level_values(1):
            data = result.xs(metric.name, level=1, axis=1)
            return data
        else:
            return None  # 未找到对应的指标

    def update_time(self):
        self.current_time_index += 1
        if self.current_time_index < len(self.time_index):
            self.current_date = self.time_index[self.current_time_index]
        else:
            self.current_date = None  # 数据结束

    def is_data_end(self):
        return self.current_time_index >= len(self.time_index)

    def get_current_time(self):
        return self.current_date
