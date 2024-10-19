
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
class Strategy:

    def __init__(self, data_handler, universe):
        self.data_handler = data_handler
        self.universe = universe

    def generate_signals(self):
        pass