from FMP.fmp import *
from FMP.main import stock_database
from Universe import Stock_Universe




class DataHandler:
    def __init__(self, universe: Stock_Universe, data_start_date: str, data_end_date: str
                 , data_requirements: [Union[Income_statement_entry
            , Balance_statement_entry, Cashflow_statement_entry
            , Financial_ratio_entry, Price_entry]]):
        self.universe = universe
        self.start_date = data_start_date
        self.end_date = data_end_date
        self.data_requirements = data_requirements
        # self.data = self.loaddata(date_requirements)
        self.current_time_index = 0
        # 使用价格数据的日期索引作为时间
        self.data = None

    def loaddata(self):

        # 这里调用外部API获取数据，返回一个dict=None
        db = stock_database(network_apikey=API_KEY)
        price_entries = list(filter(lambda x: isinstance(x, Price_entry), self.data_requirements))
        fund_entries = list(filter(lambda x: isinstance(x, (Balance_statement_entry
                                                            , Cashflow_statement_entry
                                                            , Income_statement_entry
                                                            , Financial_ratio_entry)), self.data_requirements))
        self.universe.load_data(self.start_date, self.end_date)
        if price_entries != []:
            df_price = db.get_stock_price(self.universe.get_total_universe()
                                          , fields=price_entries
                                          , end_date=self.end_date
                                          , start_date=self.start_date)
        else:
            df_price = pd.DataFrame()
        if fund_entries != []:
            dict_fund = db.get_fundamentals(self.universe.get_total_universe()
                                            , fields=fund_entries
                                            , end_date=self.end_date
                                            , start_date=self.start_date)
        else:
            dict_fund = dict()
        # 为了示例，假设data_dict已经被加载并符合要求
        data_dict = {
            'price': df_price,  # 价格数据
            'fundamental': dict_fund,  # 财务数据
            'macro': pd.DataFrame(),  # 宏观数据
        }
        # 数据预处理，处理NaN值，只进行前向填充
        # for key in data_dict:
        #     data_dict[key] = data_dict[key].ffill()

        self.data = data_dict
        self.time_index = self.data['price'].index
        self.current_date = self.time_index[self.current_time_index]

    def reset(self):
        self.current_time_index = 0
        self.current_date = self.time_index[self.current_time_index]

    def get_dynamic_universe(self,name=None):
        if name is None:
            name = "default_group"
        df = self.universe.get_universe(name)
        df = df[df.index < self.current_date]
        return df["holdings"].values.tolist()[-1]

    def get_price(self,metric:Price_entry, prev=1, allow_today_data=False):
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
        dict_fund = self.data["fundamental"]
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

        result = pd.concat(dfs,axis=1)

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