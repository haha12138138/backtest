import numpy as np
import pandas as pd
from Strategy import Strategy, rebalance

from FMP.fmp import *


class MOM_STRATEGY(Strategy):
    inherent_dependencies = Strategy.inherent_dependencies

    def __init__(self):

        super().__init__()

    @rebalance(20)
    def generate_signals(self):
        def z_score(df):
            return (df - df.mean()) / df.std()
        def calc_mom_score(series_A):
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
        def sharpe_mom(close_prices, d_ret, abs_filter= False):
            std_d_ret_6 = np.sqrt(252) * (d_ret.iloc[-120:-1].std())
            std_d_ret_12 = np.sqrt(252) * (d_ret.std())
            raw_mom_12 = (close_prices.iloc[-1] / close_prices.iloc[0]) / std_d_ret_12
            raw_mom_6 = (close_prices.iloc[-1] / close_prices.iloc[-120]) / std_d_ret_6
            mom4 = z_score(raw_mom_12)
            mom3 = z_score(raw_mom_6)
            #mom4 = raw_mom_12.rank(ascending=True)
            #mom3 = raw_mom_6.rank(ascending=True)
            comp_mom = (0.5 * mom3 + 0.5 * mom4)
            comp_mom_z = z_score(comp_mom)

            if abs_filter:
                abs_mom_idx_gt_0 = ((0.5 * raw_mom_12 + 0.5 * np.sqrt(2) * raw_mom_6) > 0).index
                return comp_mom_z.loc[abs_mom_idx_gt_0]
            else:
                return comp_mom_z

        close_prices: pd.DataFrame = self.data_handler.get_price(Price_entry.close, prev=250)
        if close_prices is None:
            # 数据不足，无法生成信号
            return None
        close_prices.dropna(axis=1, inplace=True)
        close_prices = close_prices.rolling(5).mean()
        close_prices.dropna(inplace=True)
        d_ret = close_prices.diff().dropna()
        comp_mom_z = sharpe_mom(close_prices,d_ret,abs_filter=False)
        # comp_mom = mom12
        uni = self.data_handler.get_dynamic_universe()
        mom_score = calc_mom_score(comp_mom_z[uni])
        ranked_securities = list(mom_score.rank(ascending=True).sort_values(ascending=False).index)

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

        def equal_weights(N, M):
            weights = np.array([1 for i in range(N)])
            # Normalize the weights so they sum to 1
            normalized_weights = weights / np.sum(weights)
            return normalized_weights

        M = 3
        # w = equal_weights(M, M)
        # w = geometric_weights(len(ranked_securities), M)
        w0 = mom_score.rank(ascending=True).sort_values(ascending=False)[0:M]
        w = w0/w0.sum()
        signals = {}
        for s, w1 in zip(ranked_securities[0:M], w):
            signals[s] = w1

        return signals
