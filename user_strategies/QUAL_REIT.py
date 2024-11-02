import numpy as np
import pandas as pd

from Backtester import Backtester
from FMP.fmp import (Balance_statement_entry, Financial_ratio_entry, Cashflow_statement_entry,
                     Datapoint_period)
from Strategy import Strategy, rebalance, Strategy_dependence
from Universe import Stock_Universe


class QUAL_REITS(Strategy):
    inherent_dependencies = Strategy.inherent_dependencies + [
        Strategy_dependence(Financial_ratio_entry.PB, Datapoint_period.annual, num_of_data_needed=1),
        Strategy_dependence(Financial_ratio_entry.cashflow_over_debt, Datapoint_period.annual, num_of_data_needed=4),
        Strategy_dependence(Cashflow_statement_entry.operating_cashflow, Datapoint_period.quarter,
                            num_of_data_needed=16),
        Strategy_dependence(Cashflow_statement_entry.debt_repayment, Datapoint_period.quarter, num_of_data_needed=16),
        Strategy_dependence(Cashflow_statement_entry.dividends, Datapoint_period.quarter, num_of_data_needed=16),
        Strategy_dependence(Balance_statement_entry.common_stock, Datapoint_period.quarter, num_of_data_needed=16),
        Strategy_dependence(Balance_statement_entry.preferred_stock, Datapoint_period.quarter, num_of_data_needed=16),
    ]

    def __init__(self):
        super().__init__()

    @rebalance(60)
    def generate_signals(self):
        def z_score(df: pd.Series):
            return (df - df.mean()) / df.std()

        def equal_weights(N, M):
            weights = np.array([1 for i in range(N)])
            # Normalize the weights so they sum to 1
            normalized_weights = weights / np.sum(weights)
            return normalized_weights

        operating_cashflow = self.data_handler.get_fundamentals(Cashflow_statement_entry.operating_cashflow, prev=16)
        debt_repayment = self.data_handler.get_fundamentals(Cashflow_statement_entry.debt_repayment, prev=16)
        dividends = self.data_handler.get_fundamentals(Cashflow_statement_entry.dividends, prev=16)
        common_stock = self.data_handler.get_fundamentals(Balance_statement_entry.common_stock, prev=16)
        preferred_stock = self.data_handler.get_fundamentals(Balance_statement_entry.preferred_stock, prev=16)
        cashflow_over_debt = self.data_handler.get_fundamentals(metric=Financial_ratio_entry.cashflow_over_debt, prev=4)
        PB = self.data_handler.get_fundamentals(metric=Financial_ratio_entry.PB, prev=1)
        for e in (operating_cashflow, debt_repayment, dividends, common_stock, preferred_stock, cashflow_over_debt, PB):
            if e is None:
                return None
        cashflow_over_debt.dropna(axis=1, how="all", inplace=True)
        PB.dropna(axis=1, how="all", inplace=True)
        operating_cashflow.dropna(axis=1, how="all", inplace=True)
        debt_repayment.dropna(axis=1, how="all", inplace=True)
        dividends.dropna(axis=1, how="all", inplace=True)
        common_stock.dropna(axis=1, how="all", inplace=True)
        preferred_stock.dropna(axis=1, how="all", inplace=True)

        retained_cash = operating_cashflow - debt_repayment - dividends
        total_stock = common_stock + preferred_stock
        uni = self.data_handler.get_dynamic_universe()
        uni = list((set(cashflow_over_debt.iloc[0].index))
                   .intersection(set(operating_cashflow.iloc[0].index))
                   .intersection(set(debt_repayment.iloc[0].index))
                   .intersection(set(dividends.iloc[0].index))
                   .intersection(set(common_stock.iloc[0].index))
                   .intersection(set(preferred_stock.iloc[0].index))
                   .intersection(set((PB.iloc[0] > 1).index))
                   .intersection(set(uni)))
        retained_cash_per_share_rank = z_score((retained_cash[uni].div(total_stock[uni])).median()).rank(ascending=True)
        cashflow_over_debt_rank = z_score(cashflow_over_debt[uni].median()).rank(ascending=True)
        ranked_securities = list(
            (retained_cash_per_share_rank * cashflow_over_debt_rank).sort_values(ascending=False).index)
        N = int(len(ranked_securities) / 3 + 1)
        ranked_securities = ranked_securities[0:N]
        N = int(len(ranked_securities) / 3 + 1)
        ranked_securities = PB[ranked_securities].iloc[0].rank(ascending=True).sort_values(ascending=False).index[0:N]
        w = equal_weights(N, N)
        signals = {}
        for s, w1 in zip(ranked_securities, w):
            signals[s] = w1

        return signals


if __name__ == "__main__":
    # 使用示例

    start_date = '2021-01-01'
    end_date = '2024-09-13'
    initial_cash = 10000

    uni = Stock_Universe()
    uni.add_holdings_to_group("VNQ", "index", topN=50)
    # uni.add_holdings_to_group("XLRE", "index", topN= 20)
    uni.add_benchmark("VNQ")
    backtester = Backtester(uni
                            , start_date, end_date, initial_cash
                            , strategy=QUAL_REITS)
    backtester.run_backtest()
    backtester.portfolio.position_history.set_index("Date").to_csv(
        r"..\results\REITS\high_retained_cash_high_coverage_PB.holdings.csv")
    backtester.portfolio.history.set_index("Date").to_csv(
        r"..\results\REITS\high_retained_cash_high_coverage_PB.hold_all.csv")
