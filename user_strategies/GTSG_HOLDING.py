import numpy as np
import pandas as pd

from Backtester import Backtester
from Strategy import Strategy, rebalance
from Universe import Stock_Universe


class HOLDING_STRATEGY(Strategy):
    inherent_dependencies = Strategy.inherent_dependencies

    def __init__(self):
        super().__init__()
        df = pd.read_csv(r"D:\迅雷下载\down\宏观数据\history_compact_monthly.csv")
        df["Start Date"] = pd.to_datetime(df["Start Date"], format="%m/%d/%Y")
        df["Assets"] = df["Assets"].apply(lambda x: x.split(","))
        df["Weights"] = df["Weights"].apply(lambda x: x.split(","))
        self.holdings = df
        self.last_signals = dict()

    @rebalance(1)
    def generate_signals(self):
        today = self.data_handler.current_date
        if today in self.holdings["Start Date"].values:
            row = self.holdings.loc[self.holdings["Start Date"] == today]
            assets = row["Assets"].iloc[0]
            weight = np.array(list(map(lambda x: float(x.strip('%')) / 100, row["Weights"].iloc[0])))
            weight = weight / np.sum(weight)
            assert (len(assets) == len(weight))
            signals = dict(zip(assets, weight))
            self.last_signals = signals
        else:
            signals = self.last_signals
        return signals


if __name__ == "__main__":
    # 使用示例

    start_date = '2017-01-03'
    end_date = '2024-10-24'
    initial_cash = 50000

    df = pd.read_csv(r"D:\迅雷下载\down\宏观数据\history_compact_monthly.csv")
    df["Assets"] = df["Assets"].apply(lambda x: x.split(","))
    holdings = set([e.strip(" ") for h in df["Assets"].values for e in h])
    uni = Stock_Universe()
    for h in holdings:
        uni.add_holdings_to_group(h, "stock")
    # uni.add_holdings_to_group("DGRW", "index", topN=2)
    # uni.add_holdings_to_group("XMHQ", "index", topN= 10)
    uni.add_benchmark("SPY")
    backtester = Backtester(uni
                            , start_date, end_date, initial_cash
                            , strategy=HOLDING_STRATEGY)
    backtester.run_backtest()
    backtester.portfolio.position_history.set_index("Date").to_csv("TEST.holdings.csv")
    backtester.portfolio.history.set_index("Date").to_csv("TEST.hold_all.csv")
