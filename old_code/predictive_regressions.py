"""

"""


import numpy as np
import pandas as pd
from pandasql import sqldf # for accessing pandas with SQL queries
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats import sandwich_covariance
from matplotlib import pyplot as plt
import matplotlib as mpl
import os 
os.chdir("/Users/rsigalov/Documents/PhD/disaster-risk-revision/")


# Loading data on interpolated disaster series:
combined_disaster_df = pd.read_csv("estimated_data/disaster-risk-series/combined_disaster_df.csv")
combined_disaster_df = combined_disaster_df[combined_disaster_df["days"] == 30]

# Loading data on S&P 500 returns:
ret_df = pd.read_csv("estimated_data/crsp_data/crsp_monthly_index_returns.csv")
ret_df["date"] = pd.to_datetime(ret_df["date"])
ret_df["date"] = ret_df["date"] + pd.offsets.MonthEnd(0)
ret_df["prev_month"] = ret_df["date"] + pd.offsets.MonthEnd(-1)
ret_df = ret_df[["prev_month", "sp_ret"]]
ret_df = ret_df.set_index("prev_month")

# Constructing a table for regressions:
df_for_reg = pd.pivot_table(
       combined_disaster_df, 
       index = "date", values = "value",
       columns = ["level", "var", "agg_type", "days"])

df_for_reg = pd.merge(
       df_for_reg, ret_df, left_index = True, 
       right_index = True, how = "left")

# Doing level regressions:
n_regs = len(df_for_reg.columns) - 1
df_results = pd.DataFrame(columns = ["X" ,"coef", "se", "R2"])
Y = np.array(df_for_reg["sp_ret"])

for i_col in range(n_regs):
   x_var = df_for_reg.columns[i_col]
   X = np.array(df_for_reg.iloc[:,i_col])
   X = sm.add_constant(X)
   results = sm.OLS(Y,X).fit()
   
   cov_hetero = sandwich_covariance.cov_hc1(results)
   
   dict_to_append = {"X": [x_var], 
                     "coef": [results.params[1]], 
                     "se": [np.sqrt(np.diag(cov_hetero))[1]], 
                     "R2": [results.rsquared]}
   

   df_results = df_results.append(pd.DataFrame(dict_to_append))

f = open("/Users/rsigalov/Dropbox/2019_Revision/Writing/Predictive Regressions/tables/agg_spx_pred_returns.tex", "w")
f.write(df_results.to_latex())
f.close()


# Creating indicators that market falls by 5%, 7.5% and 10% and doing
# the same regressions:
df_for_reg["falls_by_500"] = np.where(df_for_reg["sp_ret"] <= -0.05, 1, 0)
df_for_reg["falls_by_750"] = np.where(df_for_reg["sp_ret"] <= -0.075, 1, 0)
df_for_reg["falls_by_1000"] = np.where(df_for_reg["sp_ret"] <= -0.10, 1, 0)

def prob_model_est(var):
   n_regs = len(df_for_reg.columns) - 4
   df_results = pd.DataFrame(columns = ["X" ,"coef", "se", "R2"])
   Y = np.array(df_for_reg[var])
   
   for i_col in range(n_regs):
       x_var = df_for_reg.columns[i_col]
       X = np.array(df_for_reg.iloc[:,i_col])
       X = sm.add_constant(X)
       results = sm.OLS(Y,X).fit()
       
       cov_hetero = sandwich_covariance.cov_hc1(results)
       
       dict_to_append = {"X": [x_var], 
                         "coef": [results.params[1]], 
                         "se": [np.sqrt(np.diag(cov_hetero))[1]], 
                         "R2": [results.rsquared]}
       
   
       df_results = df_results.append(pd.DataFrame(dict_to_append))
       
   return df_results

f = open("/Users/rsigalov/Dropbox/2019_Revision/Writing/Predictive Regressions/tables/agg_spx_prob_model_by_500.tex", "w")
f.write(prob_model_est("falls_by_500").to_latex())
f.close()

f = open("/Users/rsigalov/Dropbox/2019_Revision/Writing/Predictive Regressions/tables/agg_spx_prob_model_by_750.tex", "w")
f.write(prob_model_est("falls_by_750").to_latex())
f.close()

f = open("/Users/rsigalov/Dropbox/2019_Revision/Writing/Predictive Regressions/tables/agg_spx_prob_model_by_1000.tex", "w")
f.write(prob_model_est("falls_by_1000").to_latex())
f.close()