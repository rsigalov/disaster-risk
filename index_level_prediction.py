"""
Index level regressions


"""

import numpy as np
import pandas as pd
from pandasql import sqldf # for accessing pandas with SQL queries
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats import sandwich_covariance

D_df = pd.read_csv("estimated_data/disaster-risk-series/D_30.csv")
ret_df = pd.read_csv("estimated_data/crsp_data/crsp_monthly_index_returns.csv")

D_df["date"] = pd.to_datetime(D_df["date"])
D_df["month_lead"] = D_df["date"] + pd.offsets.MonthEnd(1)
ret_df["date"] = pd.to_datetime(ret_df["date"])
ret_df["date"] = ret_df["date"] + pd.offsets.MonthEnd(0)

D_df = pd.merge(D_df, ret_df[["date", "vw_ret", "sp_ret"]],
                left_on ="month_lead", 
                right_on = "date", how = "left")

results = smf.ols(formula = 'sp_ret ~ D_mean', data = D_df).fit(cov_type = "HC1")
results.summary()

# Saving results:
df_results = pd.DataFrame(columns = ["Y", "X" ,"se_type", "coef", "se", "R2"])
for y_var in ["vw_ret", "sp_ret"]:
    for x_var in ["D_mean", "D_pc1"]:
        results = smf.ols(formula = y_var + ' ~ ' + x_var, data = D_df).fit()
        dict_to_append = {"Y": [y_var], "X":[x_var], "se_type": ["homo"],
                          "coef": [results.params[1]], "se": [np.array(results.bse)[1]],
                          "R2": [results.rsquared]}
        
        df_results = df_results.append(pd.DataFrame(dict_to_append))
        
        cov_hetero = sandwich_covariance.cov_hc1(results)
        dict_to_append["se_type"] = ["hetero"]
        dict_to_append["se"] = [np.sqrt(np.diag(cov_hetero))[1]]
        
        df_results = df_results.append(pd.DataFrame(dict_to_append))
        
        


