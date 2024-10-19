import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"../TEST.hold_all.csv")
df1 = pd.read_csv(r"../result/DGRW.GPOE.hold_all.csv")
df.set_index("Date",inplace=True)
df1.set_index("Date",inplace=True)
df = df.astype(float)
df1 = df1.astype(float)
df = df / df.iloc[0]
df1 = df1 / df1.iloc[0]
df1.rename(columns={"Portfolio":"DF1"},inplace=True)
df.rename(columns={"Portfolio":"DF"},inplace=True)
pd.concat([df1['DF1'],df['DF']],axis=1).plot()
plt.show()