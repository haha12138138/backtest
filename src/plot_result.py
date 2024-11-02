import os

import matplotlib.pyplot as plt
import pandas as pd

results = [
    r"../results/Fundamental_DGRW/GPOE_TTM.PEG.csv",
    r"../results/Fundamental_DGRW/GPOE_TTM.csv",
    r"../results/Fundamental_DGRW/GPOE_LYR.csv",
    r"../results/Fundamental_DGRW/GPOE_TTM.PB_PE_PEG.csv",
    r"../results/Fundamental_DGRW/GPOE_TTM.PB_PEG.csv"
]
dfs = []
for r in results:
    fname = os.path.splitext(os.path.basename(r))[0]
    df = pd.read_csv(r)
    df.set_index("Date", inplace=True)
    df.drop(["Benchmark"], axis="columns", inplace=True)
    df.rename(columns={"Portfolio": fname}, inplace=True)
    dfs.append(df)

pd.concat(dfs, axis=1).plot()
plt.show()