import os

import matplotlib.pyplot as plt
import pandas as pd

results = [
    r"../results/Fundamental_DGRW/GPOE_TTM_qoq.csv",
    r"../results/Fundamental_DGRW/GPOE_TTM.csv",
    r"../results/Fundamental_DGRW/D2A_yoy.csv",
    # r"../results/Fundamental_DGRW/GPOA_TTM.csv",
    # r"../results/Fundamental_DGRW/GROWTH1.csv",
    # r"../results/Fundamental_DGRW/GROWTH2.csv",
    # r"../results/Fundamental_DGRW/GPOE_TTM.PB_PEG.csv",
    # r"../results/Fundamental_DGRW/GPOE_TTM.PEG_PBgt1.csv"
]
# results = [
#     r"../results/Fundamental_XMHQ/GPOE_TTM.PEG_PBgt1.csv"
# ]

# results =[
# r"../results/REITS/high_cash.hold_all.csv",
# r"../results/REITS/high_cash_high_coverage.hold_all.csv",
# r"../results/REITS/high_retained_cash_high_coverage_PB.hold_all.csv"
# ]
dfs = []
start_time = "2013-01-01"
for r in results:
    fname = os.path.splitext(os.path.basename(r))[0]
    df = pd.read_csv(r)
    df.set_index("Date", inplace=True)
    df.drop(["Benchmark"], axis="columns", inplace=True)
    df.rename(columns={"Portfolio": fname}, inplace=True)
    df = df[df.index >= start_time]
    dfs.append(df / df.iloc[0])

df = pd.read_csv(results[0])
df.set_index("Date", inplace=True)
df = df[df.index >= start_time]
dfs.append(df[["Benchmark"]] / df[["Benchmark"]].iloc[0])
df = pd.concat(dfs, axis=1)
gol = (df["GPOE_TTM_qoq"] / df["D2A_yoy"])
gol_sma = gol.rolling(60).mean()
gol_lma = gol.rolling(250).mean()
pd.concat([gol, gol_sma, gol_lma], axis=1, ignore_index=True).plot()
plt.show()