import pandas as pd
import pickle
from ML_Pipeline.PreprocessPlots import PreprocessPlots
from ML_Pipeline.MLR import MLR
from ML_Pipeline.SymbolicRegression import SymbolicRegression

# Importing the data
raw_csv_data = pd.read_excel("../input/CallCenterData.xlsx")


# Check point of data
df_comp = raw_csv_data.copy()

print(df_comp.isna().sum())

df_comp.set_index("month", inplace=True)


# Setting the frequency as monthly
df_comp = df_comp.asfreq('M')

print(df_comp.isna().sum())

PreprocessPlots(df_comp)


#MLR
multipleLR = MLR()
multipleLR.run(df_comp, lag=0)
multipleLR.run(df_comp, lag=1)
multipleLR.run(df_comp, lag=2)


# Symbolic regression
symbolic_model = SymbolicRegression(df_comp)
print(symbolic_model)


# Saving the model in pickle format for future use
pickle.dump(symbolic_model,open("../output/symbolic_regression_model.pkl","wb"))
