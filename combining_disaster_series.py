"""
Created on Tue Jun  4 23:47:47 2019

@author: rsigalov
"""

import numpy as np
import pandas as pd
from pandasql import sqldf # for accessing pandas with SQL queries
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats import sandwich_covariance
from matplotlib import pyplot as plt
import matplotlib as mpl

ind_D_clamp_30 = pd.read_csv("estimated_data/disaster-risk-series/agg_D_clamp_30days.csv")
ind_D_clamp_30.columns = ["date", "pc1", "mean_filter", "mean_all"]
ind_D_clamp_30["level"] = "ind"
ind_D_clamp_30["var"] = "D_clamp"
ind_D_clamp_30["days"] = 30

ind_D_clamp_60 = pd.read_csv("estimated_data/disaster-risk-series/agg_D_clamp_60days.csv")
ind_D_clamp_60.columns = ["date", "pc1", "mean_filter", "mean_all"]
ind_D_clamp_60["level"] = "ind"
ind_D_clamp_60["var"] = "D_clamp"
ind_D_clamp_60["days"] = 60

ind_D_clamp_120 = pd.read_csv("estimated_data/disaster-risk-series/agg_D_clamp_120days.csv")
ind_D_clamp_120.columns = ["date", "pc1", "mean_filter", "mean_all"]
ind_D_clamp_120["level"] = "ind"
ind_D_clamp_120["var"] = "D_clamp"
ind_D_clamp_120["days"] = 120

ind_rnp20_30 = pd.read_csv("estimated_data/disaster-risk-series/agg_rn_prob_20mon_30days.csv")
ind_rnp20_30.columns = ["date", "pc1", "mean_filter", "mean_all"]
ind_rnp20_30["level"] = "ind"
ind_rnp20_30["var"] = "rn_prob_20mon"
ind_rnp20_30["days"] = 30

ind_rnp20_60 = pd.read_csv("estimated_data/disaster-risk-series/agg_rn_prob_20mon_60days.csv")
ind_rnp20_60.columns = ["date", "pc1", "mean_filter", "mean_all"]
ind_rnp20_60["level"] = "ind"
ind_rnp20_60["var"] = "rn_prob_20mon"
ind_rnp20_60["days"] = 60

ind_rnp20_120 = pd.read_csv("estimated_data/disaster-risk-series/agg_rn_prob_20mon_120days.csv")
ind_rnp20_120.columns = ["date", "pc1", "mean_filter", "mean_all"]
ind_rnp20_120["level"] = "ind"
ind_rnp20_120["var"] = "rn_prob_20mon"
ind_rnp20_120["days"] = 120


ind_rnp40_30 = pd.read_csv("estimated_data/disaster-risk-series/agg_rn_prob_40mon_30days.csv")
ind_rnp40_30.columns = ["date", "pc1", "mean_filter", "mean_all"]
ind_rnp40_30["level"] = "ind"
ind_rnp40_30["var"] = "rn_prob_40mon"
ind_rnp40_30["days"] = 30

ind_rnp40_60 = pd.read_csv("estimated_data/disaster-risk-series/agg_rn_prob_40mon_60days.csv")
ind_rnp40_60.columns = ["date", "pc1", "mean_filter", "mean_all"]
ind_rnp40_60["level"] = "ind"
ind_rnp40_60["var"] = "rn_prob_40mon"
ind_rnp40_60["days"] = 60

ind_rnp40_120 = pd.read_csv("estimated_data/disaster-risk-series/agg_rn_prob_40mon_120days.csv")
ind_rnp40_120.columns = ["date", "pc1", "mean_filter", "mean_all"]
ind_rnp40_120["level"] = "ind"
ind_rnp40_120["var"] = "rn_prob_40mon"
ind_rnp40_120["days"] = 120

ind_rnp2sigma_30 = pd.read_csv("estimated_data/disaster-risk-series/agg_rn_prob_2sigma_30days.csv")
ind_rnp2sigma_30.columns = ["date", "pc1", "mean_filter", "mean_all"]
ind_rnp2sigma_30["level"] = "ind"
ind_rnp2sigma_30["var"] = "rn_prob_2sigma"
ind_rnp2sigma_30["days"] = 30

ind_rnp2sigma_60 = pd.read_csv("estimated_data/disaster-risk-series/agg_rn_prob_2sigma_60days.csv")
ind_rnp2sigma_60.columns = ["date", "pc1", "mean_filter", "mean_all"]
ind_rnp2sigma_60["level"] = "ind"
ind_rnp2sigma_60["var"] = "rn_prob_2sigma"
ind_rnp2sigma_60["days"] = 60

ind_rnp2sigma_120 = pd.read_csv("estimated_data/disaster-risk-series/agg_rn_prob_2sigma_120days.csv")
ind_rnp2sigma_120.columns = ["date", "pc1", "mean_filter", "mean_all"]
ind_rnp2sigma_120["level"] = "ind"
ind_rnp2sigma_120["var"] = "rn_prob_2sigma"
ind_rnp2sigma_120["days"] = 120


ind_disaster_df = ind_D_clamp_30.copy()
ind_disaster_df = ind_disaster_df.append(ind_D_clamp_60)
ind_disaster_df = ind_disaster_df.append(ind_D_clamp_120)
ind_disaster_df = ind_disaster_df.append(ind_rnp20_30)
ind_disaster_df = ind_disaster_df.append(ind_rnp20_60)
ind_disaster_df = ind_disaster_df.append(ind_rnp20_120)
ind_disaster_df = ind_disaster_df.append(ind_rnp40_30)
ind_disaster_df = ind_disaster_df.append(ind_rnp40_60)
ind_disaster_df = ind_disaster_df.append(ind_rnp40_120)
ind_disaster_df = ind_disaster_df.append(ind_rnp2sigma_30)
ind_disaster_df = ind_disaster_df.append(ind_rnp2sigma_60)
ind_disaster_df = ind_disaster_df.append(ind_rnp2sigma_120)
ind_disaster_df["date"] = pd.to_datetime(ind_disaster_df["date"])
ind_disaster_df = pd.melt(ind_disaster_df, id_vars = ["date", "level", "var", "days"], 
                          var_name = "agg_type")
ind_disaster_df.to_csv("estimated_data/disaster-risk-series/ind_agg_disaster_df.csv", index = False)

# Loading SPX series and averaging them for each month:
spx_30 = pd.read_csv("estimated_data/disaster-risk-series/int_D_spx_days_30.csv")
spx_30["days"] = 30
spx_60 = pd.read_csv("estimated_data/disaster-risk-series/int_D_spx_days_60.csv")
spx_60["days"] = 60
spx_120 = pd.read_csv("estimated_data/disaster-risk-series/int_D_spx_days_120.csv")
spx_120["days"] = 120

spx_disaster_df = spx_30.copy()
spx_disaster_df = spx_disaster_df.append(spx_60)
spx_disaster_df = spx_disaster_df.append(spx_120)

spx_disaster_df["date"] = pd.to_datetime(spx_disaster_df["date"])
spx_disaster_df["date_adj"] = spx_disaster_df["date"] + pd.offsets.MonthEnd(0)
spx_disaster_df = spx_disaster_df.groupby(["date_adj", "days"]).mean().reset_index()
spx_disaster_df = spx_disaster_df.rename({"date_adj":"date"}, axis = 1)

spx_disaster_df = spx_disaster_df.drop(["secid", "D", "D_in_sample"], axis = 1)
spx_disaster_df["level"] = "sp_500"
spx_disaster_df["agg_type"] = "direct"
spx_disaster_df = pd.melt(spx_disaster_df, id_vars = ["date", "level", "agg_type", "days"], 
                          var_name = "var")

spx_disaster_df = spx_disaster_df[["date", "level", "var", "agg_type", "days", "value"]]
ind_disaster_df = ind_disaster_df[["date", "level", "var", "agg_type", "days", "value"]]

combined_disaster_df = spx_disaster_df.append(ind_disaster_df)

combined_disaster_df.to_csv(
        "estimated_data/disaster-risk-series/combined_disaster_df.csv", 
        index = False)

################################################################
# Comparing measures of graphs:
################################################################

# 1. 30 days, compare individual mean_all and PC1 vs. SPX D_clamp
df_comp_1 = pd.pivot_table(
        combined_disaster_df[(combined_disaster_df["days"] == 30) &
                             (combined_disaster_df["agg_type"].isin(["mean_all", "pc1", "direct"])) &
                             (combined_disaster_df["var"] == "D_clamp")], 
        index = "date", values = "value",
        columns = ["level", "var", "agg_type"])
df_comp_1.plot(figsize = (6,4))
plt.tight_layout()
plt.savefig("images/compare_disaster_series/disaster_compare_1.pdf")

# 2. 60 days, compare individual mean_all and PC1 vs. SPX D_clamp
df_comp_2 = pd.pivot_table(
        combined_disaster_df[(combined_disaster_df["days"] == 60) &
                             (combined_disaster_df["agg_type"].isin(["mean_all", "pc1", "direct"])) &
                             (combined_disaster_df["var"] == "D_clamp")], 
        index = "date", values = "value",
        columns = ["level", "var", "agg_type"])
df_comp_2.plot(figsize = (6,4))
plt.tight_layout()
plt.savefig("images/compare_disaster_series/disaster_compare_2.pdf")

# 3. 120 days, compare individual mean_all and PC1 vs. SPX D_clamp
df_comp_3 = pd.pivot_table(
        combined_disaster_df[(combined_disaster_df["days"] == 120) &
                             (combined_disaster_df["agg_type"].isin(["mean_all", "pc1", "direct"])) &
                             (combined_disaster_df["var"] == "D_clamp")], 
        index = "date", values = "value",
        columns = ["level", "var", "agg_type"])
df_comp_3.plot(figsize = (6,4))
plt.tight_layout()
plt.savefig("images/compare_disaster_series/disaster_compare_3.pdf")

# 4. mean_all D_clamp: 30 vs. 60 vs. 120 days
df_comp_4 = pd.pivot_table(
        combined_disaster_df[(combined_disaster_df["agg_type"].isin(["mean_all"])) &
                             (combined_disaster_df["var"] == "D_clamp")], 
        index = "date", values = "value",
        columns = ["level", "var", "agg_type", "days"])
df_comp_4.plot(figsize = (6,4))
plt.tight_layout()
plt.savefig("images/compare_disaster_series/disaster_compare_4.pdf")

# 5. pc1 D_clamp: 30 vs. 60 vs. 120 days
df_comp_5 = pd.pivot_table(
        combined_disaster_df[(combined_disaster_df["agg_type"].isin(["pc1"])) &
                             (combined_disaster_df["var"] == "D_clamp")], 
        index = "date", values = "value",
        columns = ["level", "var", "agg_type", "days"])
df_comp_5.plot(figsize = (6,4))
plt.tight_layout()
plt.savefig("images/compare_disaster_series/disaster_compare_5.pdf")

# 6. SPX D_clamp: 30 vs. 60 vs. 120 days
df_comp_6 = pd.pivot_table(
        combined_disaster_df[(combined_disaster_df["agg_type"].isin(["direct"])) &
                             (combined_disaster_df["var"] == "D_clamp")], 
        index = "date", values = "value",
        columns = ["level", "var", "agg_type", "days"])
df_comp_6.plot(figsize = (6,4))
plt.tight_layout()
plt.savefig("images/compare_disaster_series/disaster_compare_6.pdf")

# 7. 30 days: mean_all monthly 20% vs. pc1 monthly 20% vs. spx monthly 20%
df_comp_7 = pd.pivot_table(
        combined_disaster_df[(combined_disaster_df["agg_type"].isin(["mean_all", "pc1", "direct"])) &
                             (combined_disaster_df["var"] == "rn_prob_20mon") &
                             (combined_disaster_df["days"] == 30)], 
        index = "date", values = "value",
        columns = ["level", "var", "agg_type", "days"])
df_comp_7.plot(figsize = (6,4))
plt.tight_layout()
plt.savefig("images/compare_disaster_series/disaster_compare_7.pdf")

# Calculating correlaiton between all disaster series:
corr_all_disaster_series = pd.pivot_table(
        combined_disaster_df, values = "value", index = "date", 
        columns = ["level", "var", "agg_type", "days"]).corr()
corr_all_disaster_series.to_csv("images/compare_disaster_series/corr_all_disaster_series.csv")

# Removing 2-sigma from correlation:
f = open("/Users/rsigalov/Dropbox/2019_Revision/Writing/Predictive Regressions/tables/corr_all_disaster_series_drop_2sigma.tex", "w")
f.write(pd.pivot_table(
            combined_disaster_df[~combined_disaster_df["var"].isin(["rn_prob_2sigma", "rn_prob_2sigma"])], 
            values = "value", index = "date", 
            columns = ["level", "var", "agg_type"]).corr().round(3).to_latex())
f.close()

################################################################
# Comparing measure by calculating correlations witih the
# same groups as above
################################################################
f = open("images/compare_disaster_series/disaster_compare_corr_1.tex", "w")
f.write(df_comp_1.corr().to_latex())
f.close()

f = open("images/compare_disaster_series/disaster_compare_corr_2.tex", "w")
f.write(df_comp_2.corr().to_latex())
f.close()

f = open("images/compare_disaster_series/disaster_compare_corr_3.tex", "w")
f.write(df_comp_3.corr().to_latex())
f.close()

f = open("images/compare_disaster_series/disaster_compare_corr_4.tex", "w")
f.write(df_comp_4.corr().to_latex())
f.close()

f = open("images/compare_disaster_series/disaster_compare_corr_5.tex", "w")
f.write(df_comp_5.corr().to_latex())
f.close()

f = open("images/compare_disaster_series/disaster_compare_corr_6.tex", "w")
f.write(df_comp_6.corr().to_latex())
f.close()

f = open("images/compare_disaster_series/disaster_compare_corr_7.tex", "w")
f.write(df_comp_7.corr().to_latex())
f.close()

################################################################
# Running 1 month predictive regressions on all measures:
################################################################
combined_disaster_df = pd.read_csv("estimated_data/disaster-risk-series/combined_disaster_df.csv")
combined_disaster_df = combined_disaster_df[combined_disaster_df["days"] == 30]
# Loading data on S&P 500 returns:
ret_df = pd.read_csv("estimated_data/crsp_data/crsp_monthly_index_returns.csv")
ret_df["date"] = pd.to_datetime(ret_df["date"])
ret_df["date"] = ret_df["date"] + pd.offsets.MonthEnd(0)
ret_df["prev_month"] = ret_df["date"] + pd.offsets.MonthEnd(-1)
ret_df = ret_df[["prev_month", "sp_ret"]]
ret_df = ret_df.set_index("prev_month")

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

    
    
    