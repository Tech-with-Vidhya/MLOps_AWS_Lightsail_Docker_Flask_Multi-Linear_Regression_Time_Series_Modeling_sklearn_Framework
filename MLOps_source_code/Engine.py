import pandas as pd
from MLPipeline.PreprocessPlots import PreprocessPlots
from MLPipeline.MLR import MLR
from MLPipeline.SymbolicRegression import SymbolicRegression

# importing the data
raw_csv_data = pd.read_excel("./Input/CallCenterData.xlsx")

# check point of data
df_comp = raw_csv_data.copy()


print(df_comp.isna().sum())

df_comp.set_index("month", inplace=True)

# seeting the frequency as monthly
df_comp = df_comp.asfreq('M')

print(df_comp.isna().sum())

PreprocessPlots(df_comp)

#MLR
multipleLR = MLR()
multipleLR.run(df_comp, lag=0)
multipleLR.run(df_comp, lag=1)
multipleLR.run(df_comp, lag=2)


# Symbolic regression
SymbolicRegression(df_comp)
