from FMP.fmp import *
from FMP.main import stock_database
from Universe import Stock_Universe
import pandas as pd
import numpy as np
from Strategy import Strategy,rebalance

class TOP_N_REPLICATE(Strategy):
    def __init__(self, data_handler, universe):

        super().__init__(data_handler, universe)

    @rebalance(60)
    def generate_signals(self):
        def equal_weights(N, M):
            weights = np.array([1 for i in range(N)])
            # Normalize the weights so they sum to 1
            normalized_weights = weights / np.sum(weights)
            return normalized_weights

        #close_prices: pd.DataFrame = self.data_handler.get_price(Price_entry.close, prev=250)
        uni = self.data_handler.get_dynamic_universe()
        w = equal_weights(len(uni),None)
        signals = {}
        for s, w1 in zip(uni, w):
            signals[s] = w1

        return signals
