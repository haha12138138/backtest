import numpy as np
import pandas as pd

from Backtester import Backtester
from FMP.fmp import Income_statement_entry, Balance_statement_entry, Datapoint_period, Financial_ratio_entry, \
    Cashflow_statement_entry
from Strategy import Strategy, rebalance, Strategy_dependence
from Universe import Stock_Universe


class D2A_yoy(Strategy):
    """
    GPOE_TTM qoq + D/A yoy
    """
    inherent_dependencies = Strategy.inherent_dependencies + [
        # Strategy_dependence(Income_statement_entry.gross_profit, Datapoint_period.quarter, 5),
        # Strategy_dependence(Balance_statement_entry.total_equity, Datapoint_period.quarter, 5),
        Strategy_dependence(Financial_ratio_entry.debt_over_asset, Datapoint_period.annual, 2),
    ]

    @rebalance(60)
    def generate_signals(self):
        def equal_weights(N, M):
            weights = np.array([1 for i in range(N)])
            # Normalize the weights so they sum to 1
            normalized_weights = weights / np.sum(weights)
            return normalized_weights

        def z_score(df: pd.Series):
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

        uni = self.data_handler.get_dynamic_universe()
        # G = self.data_handler.get_fundamentals(Income_statement_entry.gross_profit,prev=5)
        # A = self.data_handler.get_fundamentals(Balance_statement_entry.total_equity,prev=5)
        D2A = self.data_handler.get_fundamentals(Financial_ratio_entry.debt_over_asset, prev=2)

        for e in (D2A,):
            if e is None:
                return None

        for e in (D2A,):
            e.dropna(axis=1, how="any", inplace=True)

        # GPOA_TTM = (G.rolling(4).sum() / A.rolling(4).mean())
        # GPOA_TTM_qoq = (GPOA_TTM.pct_change().iloc[-1]).to_frame().T
        D2A_yoy = (D2A.diff().iloc[-1] / D2A.mean()).to_frame().T

        uni = list(set(uni)
                   # .intersection(set(GPOA_TTM_qoq.iloc[0].index))
                   .intersection(set(D2A_yoy.iloc[0].index))
                   )

        if len(uni) == 0:
            return None

        # GPOA_TTM_qoq_ranked = GPOA_TTM_qoq[uni].iloc[-1].rank(ascending=True).sort_values(ascending=False)
        # D2A_yoy_ranked = D2A_yoy[uni].iloc[-1].rank(ascending=True).sort_values(ascending=False)
        # comp = GPOA_TTM_qoq_ranked * D2A_yoy_ranked
        # GPOA_TTM_qoq_z = z_score(GPOA_TTM_qoq[uni].iloc[-1])
        D2A_yoy_z = z_score(D2A_yoy[uni].iloc[-1])
        comp = D2A_yoy_z
        ranked_securities = list(comp.sort_values(ascending=False).index)
        N = int(len(ranked_securities) / 5 + 1)
        selected = ranked_securities[0:N]

        if len(selected) == 1:
            return {selected[0]: 1}
        else:
            w = msci_mom_normalize(z_score(comp[selected]))
            w = w / w.sum()
            signals = {}
            for s, w1 in zip(w.index, w.to_numpy()):
                signals[s] = w1

            return signals


class GPOE_TTM_qoq(Strategy):
    """
    GPOE_TTM qoq + D/A yoy
    """
    inherent_dependencies = Strategy.inherent_dependencies + [
        Strategy_dependence(Income_statement_entry.gross_profit, Datapoint_period.quarter, 5),
        Strategy_dependence(Balance_statement_entry.total_equity, Datapoint_period.quarter, 5),
    ]

    @rebalance(60)
    def generate_signals(self):
        def equal_weights(N, M):
            weights = np.array([1 for i in range(N)])
            # Normalize the weights so they sum to 1
            normalized_weights = weights / np.sum(weights)
            return normalized_weights

        def z_score(df: pd.Series):
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

        uni = self.data_handler.get_dynamic_universe()
        G = self.data_handler.get_fundamentals(Income_statement_entry.gross_profit, prev=5)
        A = self.data_handler.get_fundamentals(Balance_statement_entry.total_equity, prev=5)
        # D2A = self.data_handler.get_fundamentals(Financial_ratio_entry.debt_over_asset, prev=2)

        for e in (G, A):
            if e is None:
                return None

        for e in (G, A):
            e.dropna(axis=1, how="any", inplace=True)

        GPOA_TTM = (G.rolling(4).sum() / A.rolling(4).mean())
        GPOA_TTM_qoq = (GPOA_TTM.pct_change().iloc[-1]).to_frame().T
        # D2A_yoy = (D2A.diff().iloc[-1]/D2A.mean()).to_frame().T

        uni = list(set(uni)
                   .intersection(set(GPOA_TTM_qoq.iloc[0].index))
                   # .intersection(set(D2A_yoy.iloc[0].index))
                   )

        if len(uni) == 0:
            return None

        # GPOA_TTM_qoq_ranked = GPOA_TTM_qoq[uni].iloc[-1].rank(ascending=True).sort_values(ascending=False)
        # D2A_yoy_ranked = D2A_yoy[uni].iloc[-1].rank(ascending=True).sort_values(ascending=False)
        # comp = GPOA_TTM_qoq_ranked * D2A_yoy_ranked
        GPOA_TTM_qoq_z = z_score(GPOA_TTM_qoq[uni].iloc[-1])
        # D2A_yoy_z = z_score(D2A_yoy[uni].iloc[-1])
        comp = GPOA_TTM_qoq_z
        ranked_securities = list(comp.sort_values(ascending=False).index)
        N = int(len(ranked_securities) / 5 + 1)
        selected = ranked_securities[0:N]

        if len(selected) == 1:
            return {selected[0]: 1}
        else:
            w = msci_mom_normalize(z_score(comp[selected]))
            w = w / w.sum()
            signals = {}
            for s, w1 in zip(w.index, w.to_numpy()):
                signals[s] = w1

            return signals


class GROWTH2(Strategy):
    """
    GPOE_TTM qoq + D/A yoy
    """
    inherent_dependencies = Strategy.inherent_dependencies + [
        Strategy_dependence(Income_statement_entry.gross_profit, Datapoint_period.quarter, 5),
        Strategy_dependence(Balance_statement_entry.total_equity, Datapoint_period.quarter, 5),
        Strategy_dependence(Financial_ratio_entry.debt_over_asset, Datapoint_period.annual, 2),
    ]

    @rebalance(60)
    def generate_signals(self):
        def equal_weights(N, M):
            weights = np.array([1 for i in range(N)])
            # Normalize the weights so they sum to 1
            normalized_weights = weights / np.sum(weights)
            return normalized_weights

        def z_score(df: pd.Series):
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

        uni = self.data_handler.get_dynamic_universe()
        G = self.data_handler.get_fundamentals(Income_statement_entry.gross_profit, prev=5)
        A = self.data_handler.get_fundamentals(Balance_statement_entry.total_equity, prev=5)
        D2A = self.data_handler.get_fundamentals(Financial_ratio_entry.debt_over_asset, prev=2)

        for e in (G, A, D2A):
            if e is None:
                return None

        for e in (G, A, D2A):
            e.dropna(axis=1, how="any", inplace=True)

        GPOA_TTM = (G.rolling(4).sum() / A.rolling(4).mean())
        GPOA_TTM_qoq = (GPOA_TTM.pct_change().iloc[-1]).to_frame().T
        D2A_yoy = (D2A.diff().iloc[-1] / D2A.mean()).to_frame().T

        uni = list(set(uni)
                   .intersection(set(GPOA_TTM_qoq.iloc[0].index))
                   .intersection(set(D2A_yoy.iloc[0].index))
                   )

        if len(uni) == 0:
            return None

        # GPOA_TTM_qoq_ranked = GPOA_TTM_qoq[uni].iloc[-1].rank(ascending=True).sort_values(ascending=False)
        # D2A_yoy_ranked = D2A_yoy[uni].iloc[-1].rank(ascending=True).sort_values(ascending=False)
        # comp = GPOA_TTM_qoq_ranked * D2A_yoy_ranked
        GPOA_TTM_qoq_z = z_score(GPOA_TTM_qoq[uni].iloc[-1])
        D2A_yoy_z = z_score(D2A_yoy[uni].iloc[-1])
        comp = GPOA_TTM_qoq_z + D2A_yoy_z
        ranked_securities = list(comp.sort_values(ascending=False).index)
        N = int(len(ranked_securities) / 5 + 1)
        selected = ranked_securities[0:N]

        if len(selected) == 1:
            return {selected[0]: 1}
        else:
            w = msci_mom_normalize(z_score(comp[selected]))
            w = w / w.sum()
            signals = {}
            for s, w1 in zip(w.index, w.to_numpy()):
                signals[s] = w1

            return signals


class GROWTH1(Strategy):
    """
    ROE_TTM qoq+ D/A yoy
    """
    inherent_dependencies = Strategy.inherent_dependencies + [
        Strategy_dependence(Income_statement_entry.net_profit, Datapoint_period.quarter, 5),
        Strategy_dependence(Balance_statement_entry.total_equity, Datapoint_period.quarter, 5),
        Strategy_dependence(Financial_ratio_entry.debt_over_asset, Datapoint_period.annual, 2)
    ]

    @rebalance(60)
    def generate_signals(self):
        def equal_weights(N, M):
            weights = np.array([1 for i in range(N)])
            # Normalize the weights so they sum to 1
            normalized_weights = weights / np.sum(weights)
            return normalized_weights

        uni = self.data_handler.get_dynamic_universe()
        R = self.data_handler.get_fundamentals(Income_statement_entry.net_profit, prev=5)
        E = self.data_handler.get_fundamentals(Balance_statement_entry.total_equity, prev=5)
        D2A = self.data_handler.get_fundamentals(Financial_ratio_entry.debt_over_asset, prev=2)

        for e in (R, E, D2A):
            if e is None:
                return None

        for e in (R, E, D2A):
            e.dropna(axis=1, how="any", inplace=True)

        ROE_TTM = (R.rolling(4).sum() / E.rolling(4).mean())
        ROE_TTM_qoq = (ROE_TTM.pct_change().iloc[-1]).to_frame().T
        D2A_yoy = (D2A.pct_change().iloc[-1]).to_frame().T

        uni = list(set(uni)
                   .intersection(set(ROE_TTM_qoq.iloc[0].index))
                   .intersection(set(D2A_yoy.iloc[0].index))
                   )

        if len(uni) == 0:
            return None

        ROE_TTM_qoq_ranked = ROE_TTM_qoq[uni].iloc[-1].rank(ascending=True).sort_values(ascending=False)
        D2A_yoy_ranked = D2A_yoy[uni].iloc[-1].rank(ascending=True).sort_values(ascending=False)

        comp = ROE_TTM_qoq_ranked * D2A_yoy_ranked
        ranked_securities = list(comp.sort_values(ascending=False).index)
        N = int(len(ranked_securities) / 5 + 1)
        selected = ranked_securities[0:N]
        if len(selected) == 1:
            return {selected[0]: 1}
        else:
            # value_z = msci_mom_normalize(-1 * z_score(PEG[selected].iloc[0]))
            #
            signals = {}
            for s, w1 in zip(selected, equal_weights(N, N)):
                signals[s] = w1

            return signals
        pass


class GPOA(Strategy):
    inherent_dependencies = Strategy.inherent_dependencies + [
        Strategy_dependence(Income_statement_entry.gross_profit, Datapoint_period.quarter, 4),
        Strategy_dependence(Balance_statement_entry.total_assets, Datapoint_period.quarter, 4),
        # Strategy_dependence(Financial_ratio_entry.PB, Datapoint_period.quarter, num_of_data_needed=1),
        # Strategy_dependence(Financial_ratio_entry.PE, Datapoint_period.quarter, num_of_data_needed=1),
        # Strategy_dependence(Financial_ratio_entry.PEG, Datapoint_period.quarter, num_of_data_needed=1)
    ]

    def __init__(self):
        super().__init__()

    @rebalance(20)
    def generate_signals(self):
        def z_score(df: pd.Series):
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

        def equal_weights(N, M):
            weights = np.array([1 for i in range(N)])
            # Normalize the weights so they sum to 1
            normalized_weights = weights / np.sum(weights)
            return normalized_weights

        # ROE = self.data_handler.get_fundamentals(metric=Financial_ratio_entry.ROE,prev=1)
        uni = self.data_handler.get_dynamic_universe()
        GP = self.data_handler.get_fundamentals(metric=Income_statement_entry.gross_profit, prev=4)
        assets = self.data_handler.get_fundamentals(metric=Balance_statement_entry.total_assets, prev=4)
        # PB = self.data_handler.get_fundamentals(metric=Financial_ratio_entry.PB, prev=1)
        # PE = self.data_handler.get_fundamentals(metric=Financial_ratio_entry.PE, prev=1)
        # PEG = self.data_handler.get_fundamentals(metric=Financial_ratio_entry.PEG, prev=1)
        for e in (GP, assets):
            if e is None:
                return None
        GP.dropna(axis=1, how="any", inplace=True)
        assets.dropna(axis=1, how="any", inplace=True)
        # PB.dropna(axis=1, how="any", inplace=True)
        # PE.dropna(axis=1, how="any", inplace=True)
        # PEG.dropna(axis=1, how="any", inplace=True)
        GP_TTM = GP.sum().to_frame().T
        assets_TTM = assets.mean().to_frame().T

        uni = list(set(uni)
                   .intersection(set(assets_TTM.iloc[0].index))
                   .intersection(set(GP_TTM.iloc[0].index))
                   # .intersection(set((PB.iloc[0] >= 1).index))
                   # .intersection(set((PE.iloc[0]).index))
                   # .intersection(set((PEG.iloc[0]).index))
                   )

        if len(uni) == 0:
            return None
        GPOE_z = z_score((GP_TTM / assets_TTM)[uni].iloc[0]).clip(lower=-3, upper=3)

        ranked_securities = list(GPOE_z.rank(ascending=True).sort_values(ascending=False).index)
        N = int(len(ranked_securities) / 5 + 1)
        selected = ranked_securities[0:N]
        # Value_z = msci_mom_normalize(z_score(1/z_score(PB[selected.index].iloc[0]) + 1/z_score(PE[selected.index].iloc[0])))
        if len(selected) == 1:
            return {selected[0]: 1}
        else:
            # value_z = msci_mom_normalize(-1 * z_score(PEG[selected].iloc[0]))
            #
            signals = {}
            for s, w1 in zip(selected, equal_weights(N, N)):
                signals[s] = w1

            return signals
class GPOE_Value(Strategy):
    inherent_dependencies = Strategy.inherent_dependencies + [
        Strategy_dependence(Income_statement_entry.gross_profit, Datapoint_period.quarter, 4),
        Strategy_dependence(Balance_statement_entry.total_equity, Datapoint_period.quarter, 4),
        Strategy_dependence(Financial_ratio_entry.PB, Datapoint_period.quarter, num_of_data_needed=1),
        # Strategy_dependence(Financial_ratio_entry.PE, Datapoint_period.quarter, num_of_data_needed=1),
        Strategy_dependence(Financial_ratio_entry.PEG, Datapoint_period.quarter, num_of_data_needed=1)
    ]

    def __init__(self):
        super().__init__()

    @rebalance(20)
    def generate_signals(self):
        def z_score(df: pd.Series):
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
        def equal_weights(N, M):
            weights = np.array([1 for i in range(N)])
            # Normalize the weights so they sum to 1
            normalized_weights = weights / np.sum(weights)
            return normalized_weights

        # ROE = self.data_handler.get_fundamentals(metric=Financial_ratio_entry.ROE,prev=1)
        uni = self.data_handler.get_dynamic_universe()
        GP = self.data_handler.get_fundamentals(metric=Income_statement_entry.gross_profit, prev=4)
        equity = self.data_handler.get_fundamentals(metric=Balance_statement_entry.total_equity, prev=4)
        PB = self.data_handler.get_fundamentals(metric=Financial_ratio_entry.PB, prev=1)
        # PE = self.data_handler.get_fundamentals(metric=Financial_ratio_entry.PE, prev=1)
        PEG = self.data_handler.get_fundamentals(metric=Financial_ratio_entry.PEG, prev=1)
        for e in (GP, equity, PB, PEG):
            if e is None:
                return None
        GP.dropna(axis=1, how="any", inplace=True)
        equity.dropna(axis=1, how="any", inplace=True)
        PB.dropna(axis=1, how="any", inplace=True)
        # PE.dropna(axis=1, how="any", inplace=True)
        PEG.dropna(axis=1, how="any", inplace=True)
        GP_TTM = GP.sum().to_frame().T
        equity_TTM = equity.mean().to_frame().T

        uni = list(set(uni)
                   .intersection(set(equity_TTM.iloc[0].index))
                   .intersection(set(GP_TTM.iloc[0].index))
                   # .intersection(set((PB.iloc[0] >= 1).index))
                   # .intersection(set((PE.iloc[0]).index))
                   .intersection(set((PEG.iloc[0]).index))
                   )

        if len(uni) == 0:
            return None
        GPOE_z = z_score((GP_TTM / equity_TTM)[uni].iloc[0]).clip(lower=-3, upper=3)

        ranked_securities = list(GPOE_z.rank(ascending=True).sort_values(ascending=False).index)
        N = int(len(ranked_securities) / 5 + 1)
        selected = ranked_securities[0:N]
        # Value_z = msci_mom_normalize(z_score(1/z_score(PB[selected.index].iloc[0]) + 1/z_score(PE[selected.index].iloc[0])))
        if len(selected) == 1:
            return {selected[0]: 1}
        else:
            # value_z = msci_mom_normalize(-1 * z_score(PEG[selected].iloc[0]))
            #
            signals = {}
            for s, w1 in zip(selected, equal_weights(N, N)):
                signals[s] = w1

            return signals
class GPOE_LYR(Strategy):
    inherent_dependencies = Strategy.inherent_dependencies + [
        Strategy_dependence(Income_statement_entry.gross_profit, Datapoint_period.annual, 1),
        Strategy_dependence(Balance_statement_entry.total_equity, Datapoint_period.annual, 1),
        # Strategy_dependence(Financial_ratio_entry.PB,Datapoint_period.quarter,num_of_data_needed=1),
        # Strategy_dependence(Financial_ratio_entry.PEG,Datapoint_period.quarter,num_of_data_needed=1)
    ]

    def __init__(self):
        super().__init__()

    @rebalance(60)
    def generate_signals(self):
        def z_score(df: pd.Series):
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

        def equal_weights(N, M):
            weights = np.array([1 for i in range(N)])
            # Normalize the weights so they sum to 1
            normalized_weights = weights / np.sum(weights)
            return normalized_weights

        # ROE = self.data_handler.get_fundamentals(metric=Financial_ratio_entry.ROE,prev=1)
        GP = self.data_handler.get_fundamentals(metric=Income_statement_entry.gross_profit, prev=1)
        equity = self.data_handler.get_fundamentals(metric=Balance_statement_entry.total_equity, prev=1)
        # PB = self.data_handler.get_fundamentals(metric=Financial_ratio_entry.PB, prev=1)
        # PEG = self.data_handler.get_fundamentals(metric=Financial_ratio_entry.PEG,prev=1)
        for e in (GP, equity):
            if e is None:
                return None
        GP.dropna(axis=1, how="any", inplace=True)
        equity.dropna(axis=1, how="any", inplace=True)
        # PB.dropna(axis=1, how="all", inplace=True)
        # PEG.dropna(axis=1, how="all", inplace=True)
        GP_TTM = GP
        equity_TTM = equity

        uni = self.data_handler.get_dynamic_universe()
        uni = list(set(uni)
                   .intersection(set(equity_TTM.iloc[0].index))
                   .intersection(set(GP_TTM.iloc[0].index))
                   # .intersection(set((PB.iloc[0] >1).index))
                   # .intersection(set((PEG.iloc[0] >0).index))
                   )

        if len(uni) == 0:
            return None
        GPOE_z = z_score((GP_TTM / equity_TTM)[uni].iloc[0]).clip(lower=-3, upper=3)

        ranked_securities = list(GPOE_z.rank(ascending=True).sort_values(ascending=False).index)
        N = int(len(ranked_securities) / 5 + 1)
        selected = ranked_securities[0:N]
        # Value_z = msci_mom_normalize(z_score(1/z_score(PB[selected.index].iloc[0]) + 1/z_score(PE[selected.index].iloc[0])))
        # value_z = msci_mom_normalize(z_score(1/(PEG[selected.index].iloc[0])))
        signals = {}
        for s, w1 in zip(ranked_securities[0:N], equal_weights(N, N)):
            signals[s] = w1

        return signals

if __name__ == "__main__":
    start_date = '2014-01-01'
    end_date = '2024-11-07'
    initial_cash = 10000
    uni = Stock_Universe()
    uni.add_holdings_to_group("DGRW", "index", topN=30)
    uni.add_benchmark("DGRW")
    strategy = D2A_yoy
    print(strategy.__name__)
    backtester = Backtester(uni
                            , start_date, end_date, initial_cash
                            , strategy=strategy)
    backtester.run_backtest()
    backtester.portfolio.position_history.set_index("Date").to_csv(
        f"../results/Fundamental_DGRW/{strategy.__name__}.holdings.csv")
    backtester.portfolio.history.set_index("Date").to_csv(f"../results/Fundamental_DGRW/{strategy.__name__}.csv")
