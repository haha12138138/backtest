import pandas as pd

from fmp import *

class Stock_Universe:
    def __init__(self):
        self.__requirement = {'default_group':[]}
        self.__groups ={"default_group": None}
        self.__total_universe = set()

    def add_holdings_to_group(self, ticker, type, topN=1, group_name='default_group'):
        if group_name in self.__requirement.keys():
            self.__requirement[group_name].append((ticker,type,topN))
        else:
            self.__requirement[group_name] = []
            self.__requirement[group_name].append((ticker, type, topN))

    def add_benchmark(self,ticker):
        self.add_holdings_to_group(ticker, type="stock", group_name="benchmark")

    def load_data(self,start_date, end_date):
        for name, assets_info in self.__requirement.items():
            dfs = []
            individual_stocks = []
            for asset in assets_info:
               ticker , asset_type, topN = asset
               if asset_type == "index":
                   df = fmp_get_etf_holdings(ticker, apikey=API_KEY
                                        , start_date=start_date, end_date=end_date, highest_N=topN)
                   dfs.append(df)
               else:
                   individual_stocks.append(ticker)
            if dfs == []:
                merged=pd.DataFrame(index=[pd.to_datetime(start_date)])
                row_num = 1
            else:
                merged = pd.concat(dfs,axis=1).ffill().apply(lambda s: s.fillna({i: [] for i in df.index}))
                row_num = len(merged)
            merged["const"] = [individual_stocks for _ in range(row_num)]
            self.__groups[name]= pd.DataFrame(merged.apply(lambda row: sum(row, []), axis=1),columns=['holdings'])
            self.__total_universe = self.__total_universe.union(set(self.__groups[name].sum().values[0]))
        return 0

    def get_universe(self,name):
        """
        get the holding data of a group.
        should only be called by the data_handler
        :param name: name of the group
        :return: Error or the data:pd.DataFrame
        """
        if name not in self.__groups.keys():
            # raise ValueError(f"{name} is not among the group names: {list(self.__groups.keys())}")
            return None
        else:
            return self.__groups[name]

    def get_total_universe(self):
        return self.__total_universe