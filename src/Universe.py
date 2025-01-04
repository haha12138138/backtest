from FMP.fmp import *


class Stock_Universe:
    """
       A class to manage and organize a universe of stocks based on different groups and requirements.
       The class allows adding stocks or ETFs to specific groups, loading data for these groups, and retrieving the universe of stocks.
    """
    def __init__(self):
        """
           A class to manage and organize a universe of stocks based on different groups and requirements.
           The class allows adding stocks or ETFs to specific groups, loading data for these groups, and retrieving the universe of stocks.
        """
        self.__requirement = {'default_group': []}
        self.__groups = {"default_group": None}
        self.__total_universe = set()

    def add_holdings_to_group(self, ticker, type, topN=1, group_name='default_group'):
        """
                Adds a stock or ETF to a specific group with a given type and topN value.

                :param ticker: The ticker symbol of the stock or ETF.
                :param type: The type of the asset (e.g., "stock", "index").
                :param topN: The number of top holdings to consider (default is 1).
                :param group_name: The name of the group to which the asset is added (default is 'default_group').
        """
        if group_name in self.__requirement.keys():
            self.__requirement[group_name].append((ticker, type, topN))
        else:
            self.__requirement[group_name] = []
            self.__requirement[group_name].append((ticker, type, topN))

    def add_benchmark(self, ticker):
        """
                Adds a benchmark stock to the 'benchmark' group.

                :param ticker: The ticker symbol of the benchmark stock.
        """
        self.add_holdings_to_group(ticker, type="stock", group_name="benchmark")

    def load_data(self, start_date, end_date):
        """
                Loads data for all groups based on the specified start and end dates.

                :param start_date: The start date for data retrieval in "YYYY-MM-DD" format.
                :param end_date: The end date for data retrieval in "YYYY-MM-DD" format.
                :return: 0 indicating successful data loading.
        """
        for name, assets_info in self.__requirement.items():
            dfs = []
            individual_stocks = []
            for asset in assets_info:
                ticker, asset_type, topN = asset
                if asset_type == "index":
                    df = fmp_get_etf_holdings(ticker, apikey=API_KEY
                                              , start_date=(
                                datetime.datetime.strptime(start_date, "%Y-%m-%d") - datetime.timedelta(
                            1)).strftime("%Y-%m-%d"), end_date=end_date, highest_N=topN)
                    dfs.append(df)
                else:
                    individual_stocks.append(ticker)
            if dfs == []:
                merged = pd.DataFrame(index=[pd.to_datetime(start_date)])
                row_num = 1
            else:
                merged = pd.concat(dfs, axis=1).ffill().apply(lambda s: s.fillna({i: [] for i in df.index}))
                row_num = len(merged)
            merged["const"] = [individual_stocks for _ in range(row_num)]
            self.__groups[name] = pd.DataFrame(merged.apply(lambda row: sum(row, []), axis=1), columns=['holdings'])
            self.__total_universe = self.__total_universe.union(set(self.__groups[name].sum().values[0]))
        return 0

    def get_universe(self, name):
        """
        Retrieves the holdings data for a specific group.

        :param name: The name of the group.
        :return: The holdings data as a pandas DataFrame, or None if the group does not exist.
        """
        if name not in self.__groups.keys():
            # raise ValueError(f"{name} is not among the group names: {list(self.__groups.keys())}")
            return None
        else:
            return self.__groups[name]

    def get_total_universe(self):
        """
                Retrieves the total universe of stocks across all groups.

                :return: A set containing all the stocks in the universe.
        """
        return self.__total_universe
