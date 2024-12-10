import numpy as np
import pandas as pd
import statsmodels.api as sm

from Backtester import Backtester
from FMP.fmp import *
from Strategy import Strategy, rebalance, Strategy_dependence
from Universe import Stock_Universe


class MOM_STRATEGY(Strategy):
    inherent_dependencies = Strategy.inherent_dependencies + [
        Strategy_dependence(Price_entry.adjClose, 'd', 120)
    ]

    def __init__(self):

        super().__init__()

    @rebalance(20)
    def generate_signals(self):
        def z_score(df):
            return (df - df.mean()) / df.std()
        def msci_mom_normalize(series_A):
            # Step 1: Create Series B based on the rules given for Series A
            series_B = series_A  # series_A.clip(lower=-3, upper=3)

            # Step 2: Apply the transformation rules for Series B
            def update_value(Z):
                if Z > 0:
                    return 1 + Z
                elif Z < 0:
                    return 1 / (1 - Z)
                else:
                    return Z  # If Z == 0, it remains 0

            series_B_transformed = series_B.apply(update_value)

            return series_B_transformed
        def sharpe_mom(prices):
            d_ret = prices.diff().dropna()
            std = d_ret.std()
            raw_mom = prices.iloc[-1] / prices.iloc[0]
            return raw_mom / std
        def resid_mom(asset_prices, mkt_prices):
            log_d_ret_mkt = np.log((mkt_prices / mkt_prices.shift(1)).dropna())
            log_d_ret_asset = np.log((asset_prices / asset_prices.shift(1)).dropna())

            # 2. 定义回归函数，返回残差
            def get_residuals(y, X):
                # X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()
                return model.resid
            # 3. 对每个资产的收益率进行回归，得到残差
            residuals = log_d_ret_asset.apply(lambda y: get_residuals(y, log_d_ret_mkt), axis=0)
            # 4. 计算残差动量：残差的均值除以标准差
            residual_momentum = residuals.apply(lambda r: np.sum(r) / np.std(r), axis=0)
            # 5. 将结果转换为 DataFrame，只有一行，代表最新截面动量值

            return residual_momentum

        close_prices: pd.DataFrame = self.data_handler.get_price(Price_entry.close, prev=120)
        if close_prices is None:
            # 数据不足，无法生成信号
            return None
        close_prices.dropna(axis=1, inplace=True)
        close_prices = close_prices.rolling(5).mean()
        close_prices.dropna(inplace=True)
        asset_uni = self.data_handler.get_dynamic_universe()
        mkt_uni = self.data_handler.get_dynamic_universe(name="benchmark")
        asset_uni = list(set(asset_uni).intersection(set(close_prices.iloc[0].index)))
        mom = resid_mom(close_prices[asset_uni], close_prices[mkt_uni])
        # mom = sharpe_mom(close_prices[asset_uni])
        # mom_norm = msci_mom_normalize(z_score(mom))
        ranked_securities = list(mom.rank(ascending=True).sort_values(ascending=False).index)
        M = 6
        def geometric_weights(N, M):
            """
            Calculate geometric weights for M selected stocks from a total of N stocks.
            Parameters:
            N (int): Total number of stocks.
            M (int): Number of stocks to select for weighting.
            Returns:
            list: Geometric weights for the selected M stocks.
            """
            # Ensure M <= N
            if M > N:
                # raise ValueError("M cannot be greater than N.")
                M = N
            # Define the common ratio for geometric progression
            # Usually, a ratio less than 1 is used, e.g., 0.5 for halving weights
            ratio = 0.5
            # Generate the geometric progression weights before normalization
            weights = np.array([ratio ** i for i in range(M)])
            # Normalize the weights so they sum to 1
            normalized_weights = weights / np.sum(weights)
            return normalized_weights

        w = geometric_weights(len(ranked_securities), M)
        signals = {}
        for s, w1 in zip(ranked_securities[0:M], w):
            signals[s] = w1

        return signals

if __name__ == "__main__":
    # 使用示例

    start_date = '2019-06-01'
    end_date = '2024-10-13'
    initial_cash = 10000

    uni = Stock_Universe()
    uni.add_holdings_to_group("XLK", "index", topN=10)
    uni.add_holdings_to_group("XLE", "index", topN=10)
    uni.add_holdings_to_group("XLV", "index", topN=10)
    # uni.add_holdings_to_group("XLRE", "index", topN= 20)
    uni.add_benchmark("SPY")
    backtester = Backtester(uni
                            , start_date, end_date, initial_cash
                            , strategy=MOM_STRATEGY)
    backtester.run_backtest()
    backtester.portfolio.position_history.set_index("Date").to_csv(
        r"..\results\ETF_ROTATION\XLK_XLE_XLV.resid_no_a.30pick6.holdings.csv")
    backtester.portfolio.history.set_index("Date").to_csv(r"..\results\ETF_ROTATION\XLK_XLE_XLV.resid_no_a.30pick6.csv")
