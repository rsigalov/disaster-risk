"""
Index level regressions

Running aggregate level predictive regressions using D-measure
"""

import numpy as np
import pandas as pd
from pandasql import sqldf # for accessing pandas with SQL queries
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats import sandwich_covariance
from matplotlib import pyplot as plt

D_df = pd.read_csv("estimated_data/disaster-risk-series/combined_disaster_df.csv")
#D_df[(D_df["level"] == "ind") &
#     (D_df["days"] == 30) &
#     (D_df["agg_type"] == "mean_all") &
#     (D_df["var"] == "D_clamp")]
D_df = pd.pivot_table(
        D_df, 
        index = "date", values = "value",
        columns = ["level", "var", "agg_type", "days"])

ret_df = pd.read_csv("estimated_data/crsp_data/crsp_monthly_index_returns.csv")

D_df["date"] = pd.to_datetime(D_df["date"])
D_df["month_lead"] = D_df["date"] + pd.offsets.MonthEnd(1)
ret_df["date"] = pd.to_datetime(ret_df["date"])
ret_df["date"] = ret_df["date"] + pd.offsets.MonthEnd(0)

D_df = pd.merge(D_df, ret_df[["date", "vw_ret", "sp_ret"]],
                left_on ="month_lead", 
                right_on = "date", how = "left")

results = smf.ols(formula = 'sp_ret ~ D_mean_all', data = D_df).fit(cov_type = "HC1")
results.summary()

# Saving results:
df_results = pd.DataFrame(columns = ["Y", "X" ,"se_type", "coef", "se", "R2"])
for y_var in ["vw_ret", "sp_ret"]:
    for x_var in ["D_mean_all", "D_pc1", "D_mean_filter"]:
        results = smf.ols(formula = y_var + ' ~ ' + x_var, data = D_df).fit()
        dict_to_append = {"Y": [y_var], "X":[x_var], "se_type": ["homo"],
                          "coef": [results.params[1]], "se": [np.array(results.bse)[1]],
                          "R2": [results.rsquared]}
        
        df_results = df_results.append(pd.DataFrame(dict_to_append))
        
        cov_hetero = sandwich_covariance.cov_hc1(results)
        dict_to_append["se_type"] = ["hetero"]
        dict_to_append["se"] = [np.sqrt(np.diag(cov_hetero))[1]]
        
        df_results = df_results.append(pd.DataFrame(dict_to_append))
        
df_results
       
####################################################################################
# Comparing D and probabilities aggregate series
#################################################################################### 
D_df = pd.read_csv("estimated_data/disaster-risk-series/D_30.csv")
ret_df = pd.read_csv("estimated_data/crsp_data/crsp_monthly_index_returns.csv")

D_df["date"] = pd.to_datetime(D_df["date"])
prob_df = pd.read_csv("estimated_data/disaster-risk-series/agg_prob_20perc_30days.csv")
prob_df = prob_df.rename({"D_pc1": "prob_pc1", 
                          "D_mean_filter": "prob_mean_filter",
                          "D_mean_all": "prob_mean_all",}, axis = 1)
        
prob_df["date"] = pd.to_datetime(prob_df["date"])
all_df = pd.merge(prob_df, D_df, on = "date", how = "inner")

all_df.set_index("date").plot(figsize = (8,6))
all_df.set_index("date").corr()

####################################################################################
# Doing the same but with probabilities:
####################################################################################
prob_df = pd.read_csv("estimated_data/disaster-risk-series/agg_prob_20perc_30days.csv")
prob_df = prob_df.rename({"D_pc1": "prob_pc1", 
                          "D_mean_filter": "prob_mean_filter",
                          "D_mean_all": "prob_mean_all",}, axis = 1)
        
ret_df = pd.read_csv("estimated_data/crsp_data/crsp_monthly_index_returns.csv")
        
prob_df["date"] = pd.to_datetime(prob_df["date"])
prob_df["month_lead"] = prob_df["date"] + pd.offsets.MonthEnd(1)
ret_df["date"] = pd.to_datetime(ret_df["date"])
ret_df["date"] = ret_df["date"] + pd.offsets.MonthEnd(0)

prob_df = pd.merge(prob_df, ret_df[["date", "vw_ret", "sp_ret"]],
                left_on ="month_lead", 
                right_on = "date", how = "left")

rhs_var = "prob_mean_all"
cutoff_list = [-0.1, -0.09, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0]
coef_list = []
se_list = []
r2_list = []

for x in cutoff_list:
    prob_df["dummy_decline"] = np.where(prob_df["vw_ret"] <= x, 1, 0)
    
    results = smf.ols(formula = 'dummy_decline ~ ' + rhs_var, 
                      data = prob_df).fit(cov_type = "HC1")
    coef_list.append(results.params[1])
    se_list.append(results.HC1_se[1])
    r2_list.append(results.rsquared)
#    print("%.4f, %.4f, %.4f, %.4f" % (x, results.params[1], results.HC1_se[1], results.rsquared))
    

fig, ax = plt.subplots(figsize = (7,4))
ax.errorbar(cutoff_list, coef_list, xerr = 0,
            yerr=[x*1.96 for x in se_list],
            fmt='-o')
ax.axhline(0, color = "black", linewidth = 2, alpha = 0.7)
ax.set_xlabel("x")
ax.set_title("beta from $1\{ret_{m,t+1} < x\} = alpha + beta \cdot \overline{P_t^Q(ret_{i,t+1} < -0.2)} + u_{t+1}$")
    












