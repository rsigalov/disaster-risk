import numpy as np
import pandas as pd
from pandasql import sqldf # for accessing pandas with SQL queries
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats import sandwich_covariance
from matplotlib import pyplot as plt
import matplotlib as mpl

# Loading estimated disaster series:
disaster_df = pd.read_csv(
        "estimated_data/final_regression_dfs/disaster_sort_ret.csv")
disaster_df = disaster_df.rename({"month_lead": "date"}, axis = 1)
ret_df = pd.read_csv("estimated_data/crsp_data/crsp_monthly_returns.csv")

# Dealing with dates:
disaster_df["date"] = pd.to_datetime(disaster_df["date"]) + pd.offsets.MonthEnd(0)
ret_df["date"] = pd.to_datetime(ret_df["date"]) + pd.offsets.MonthEnd(0)

# Merging datasets:
short_disaster_df = disaster_df[(disaster_df["level"] == "ind") & 
            (disaster_df["var"] == "rn_prob_20mon") & 
            (disaster_df["agg_type"] == "pc1") & 
            (disaster_df["days"] == 30)][["date", "value"]].rename({"value": "p"}, axis = 1)

reg_df = pd.merge(ret_df, short_disaster_df, on = "date")

# Claculating exposure for each company (first without filter):
name_list = []
beta_list = []
mean_ret_list = []

reg_df_short = reg_df[reg_df.date <= "2005-12-31"]

for (name, sub_df) in reg_df.groupby("permno"):
    if sub_df.shape[0] > 150:
        results = smf.ols(formula = 'ret ~ p', data = sub_df).fit()
        name_list.append(name)
        beta_list.append(results.params[1])
        mean_ret_list.append(np.mean(sub_df.ret))
        
beta_mean_df = pd.DataFrame({"permno": name_list, 
                             "beta": beta_list,
                             "mean_ret": mean_ret_list})
    
plt.scatter(beta_mean_df.beta, beta_mean_df.mean_ret)

results = smf.ols(formula = 'mean_ret ~ beta', data = beta_mean_df).fit()
results.summary()

# Sorting into 10 portfolios based on their beta and calculating average return 
# within each portfolios:
beta_mean_df = beta_mean_df.sort_values("beta")
port_beta_list = []
port_ret_list = []
num_ports = 10
comps_in_port = int(beta_mean_df.shape[0]/num_ports)

for i_port in range(num_ports):
    sub_df = beta_mean_df.iloc[i_port*comps_in_port:(i_port+1)*comps_in_port]
    port_beta_list.append(np.mean(sub_df.beta))
    port_ret_list.append(np.mean(sub_df.mean_ret))

plt.scatter(port_beta_list, port_ret_list)





