import Universe

if __name__ == "__main__":
    uni = Universe.Stock_Universe()
    # uni.add_holdings_to_group("DGRW", "index", 5)
    # uni.add_holdings_to_group("XMHQ", "index", 5)
    uni.add_holdings_to_group("MSFT", "stock")
    uni.add_holdings_to_group("SPY", "stock", group_name="benchmark")
    uni.load_data(start_date="2020-01-01", end_date="2024-10-12")
    print("HAHA")
