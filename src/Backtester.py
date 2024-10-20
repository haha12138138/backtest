#import deprecation

from Datahandler import DataHandler
from FMP.fmp import Price_entry
from Portfolio import Portfolio
from Universe import Stock_Universe


class Backtester:
    def __init__(self, universe: Stock_Universe, start_date: str, end_date: str, initial_cash: int, strategy):
        self.strategy = strategy()
        self.data_handler = DataHandler(universe
                                        , start_date, end_date, strategy)
        self.data_handler.loaddata()
        self.strategy.add_data_handler(self.data_handler)
        self.universe = universe
        self.benchmark = universe.get_universe("benchmark")
        self.portfolio = Portfolio(initial_cash, universe, self.benchmark)

    def run_backtest(self):
        while not self.data_handler.is_data_end():
            current_time = self.data_handler.get_current_time()
            print(f"回测日期: {current_time}")

            # 获取策略信号
            signals = self.strategy.generate_signals()

            open_prices_df = self.data_handler.get_price(Price_entry.open, prev=1, allow_today_data=True)
            close_prices_df = self.data_handler.get_price(Price_entry.close, prev=1, allow_today_data=True)
            adjclose_prices_df = self.data_handler.get_price(Price_entry.adjClose, prev=1, allow_today_data=True)
            adjopen_prices_df = (open_prices_df / close_prices_df) * adjclose_prices_df
            adjopen_prices = adjopen_prices_df[list(self.universe.get_total_universe())].iloc[-1].to_dict()
            if self.benchmark is not None:
                bench_adjopen_prices = adjopen_prices_df[(self.benchmark.iloc[0].values)[0][0]].iloc[-1]
            else:
                bench_adjopen_prices = None

            if adjopen_prices_df is None:
                # 无法获取开盘价，跳过当前日期
                self.data_handler.update_time()
                continue

            if signals is not None:
                # 获取当日开盘价
                # 下订单
                self.portfolio.order_to_percentage(signals, adjopen_prices)
                # 更新投资组合
                self.portfolio.update_portfolio(self.data_handler.get_current_time(), adjopen_prices,
                                                bench_adjopen_prices)

            # 时间推进
            self.data_handler.update_time()

        # 绘制净值曲线
        self.portfolio.plot_nav()
