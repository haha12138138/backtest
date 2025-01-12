from dataclasses import dataclass

from FMP.fmp import *
from fmp_datatypes import Datapoint_period


def rebalance(rebalance_period):
    def decorator(func):
        temp = rebalance_period
        i = 0
        last_signal = None

        def wrapper(*args, **kwargs):
            nonlocal temp, i, last_signal
            if i % temp == 0:
                signal = func(*args, **kwargs)
                last_signal = signal
            else:
                signal = last_signal
            i += 1
            return signal

        return wrapper

    return decorator


@dataclass
class Strategy_dependence:
    dependence_name: Union[
        Price_entry, Financial_ratio_entry, Balance_statement_entry, Cashflow_statement_entry, Income_statement_entry]
    period: Datapoint_period
    num_of_data_needed: int

    def __init__(self, dependence_name, period, num_of_data_needed):
        self.dependence_name = dependence_name
        self.period = period
        self.num_of_data_needed = num_of_data_needed


class Strategy:
    inherent_dependencies = [Strategy_dependence(Price_entry.open, 'd', 1),
                             Strategy_dependence(Price_entry.adjClose, 'd', 1),
                             Strategy_dependence(Price_entry.close, 'd', 1)]

    def __init__(self):
        self.data_handler = None

    def add_data_handler(self, data_handler):
        self.data_handler = data_handler

    def generate_signals(self):
        pass
