"""
Script for merging disaster series with monthly returns data
"""

import numpy as np
import pandas as pd
from pandasql import sqldf # for accessing pandas with SQL queries
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats import sandwich_covariance
from statsmodels.iolib.summary2 import summary_col # For summarizing regression results

########################################################################
file_name = "estimated_data/interpolated_D/int_D_clamp_days_60.csv"
var = "D_clamp" # variable to look for non-missing observations
min_obs_in_month = 10
min_share_month = 0

# Loading data on interpolated values
D_df = pd.read_csv(file_name)

# Dealing with dates:
D_df["date"] = pd.to_datetime(D_df["date"])
D_df["date_adj"] = D_df["date"] + pd.offsets.MonthEnd(0)
D_df = D_df.drop("date", axis = 1)


########################################################################
# Limiting to companies with at least 15 observations in a month in at least 80%
# months in the sample from January 1996 to December 2017.
def min_month_obs(x):
    return x[var].count() > min_obs_in_month

D_filter_1 = D_df.groupby(["secid", "date_adj"]).filter(min_month_obs)
D_mon_mean = D_filter_1.groupby(["secid", "date_adj"]).mean().reset_index()

num_months = len(np.unique(D_mon_mean["date_adj"]))
def min_sample_obs(x):
    return x[var].count() > num_months * min_share_month

D_filter = D_mon_mean.groupby("secid").filter(min_sample_obs)


########################################################################
# Loading data on monthly return and linking data:
ret_df = pd.read_csv("estimated_data/crsp_data/crsp_monthly_returns.csv")
oclink = pd.read_csv("estimated_data/crsp_data/optionmetrics_crsp_link.csv")

# Getting the best link for each month end of D-clamp:
oclink = oclink[oclink.score < 6]
oclink["sdate"] = [str(int(x)) for x in oclink["sdate"]]
oclink["sdate"] = pd.to_datetime(oclink["sdate"], format = "%Y%m%d")
oclink["edate"] = [str(int(x)) for x in oclink["edate"]]
oclink["edate"] = pd.to_datetime(oclink["edate"], format = "%Y%m%d")

pysqldf = lambda q: sqldf(q, globals())

q1 = """
select 
    d.*,
    s1.permno as permno_1, 
    s2.permno as permno_2,
    s3.permno as permno_3,
    s4.permno as permno_4,
    s5.permno as permno_5
from D_filter as d
left join (
    select 
        secid, permno, sdate, edate
    from oclink
    where score = 1
) as s1
on d.secid = s1.secid
and d.date_adj >= s1.sdate
and d.date_adj <= s1.edate
left join (
    select 
        secid, permno, sdate, edate
    from oclink
    where score = 2
) as s2
on d.secid = s2.secid
and d.date_adj >= s2.sdate
and d.date_adj <= s2.edate
left join (
    select 
        secid, permno, sdate, edate
    from oclink
    where score = 3
) as s3
on d.secid = s3.secid
and d.date_adj >= s3.sdate
and d.date_adj <= s3.edate
left join (
    select 
        secid, permno, sdate, edate
    from oclink
    where score = 4
) as s4
on d.secid = s4.secid
and d.date_adj >= s4.sdate
and d.date_adj <= s4.edate
left join (
    select 
        secid, permno, sdate, edate
    from oclink
    where score = 5
) as s5
on d.secid = s5.secid
and d.date_adj >= s5.sdate
and d.date_adj <= s5.edate
"""

tmp = pysqldf(q1)

# Filtering and providing the best match:
q2 = """
select 
    *,
    COALESCE(permno_1, permno_2, permno_3, permno_4, permno_5) as permno
from tmp
"""

disaster_ret_df = pysqldf(q2)
disaster_ret_df = disaster_ret_df.drop(
        ["permno_1", "permno_2", "permno_3", "permno_4", "permno_5"], axis = 1)

# Merging with returns next month:
disaster_ret_df = disaster_ret_df.rename({"date_adj": "date"}, axis = 1)
disaster_ret_df["date"] = pd.to_datetime(disaster_ret_df["date"])
disaster_ret_df["month_lead"] = disaster_ret_df["date"] + pd.offsets.MonthEnd(1)
disaster_ret_df = disaster_ret_df.drop("date", axis = 1)

ret_df["date"] = pd.to_datetime(ret_df["date"])
ret_df["date"] = ret_df["date"] + pd.offsets.MonthEnd(0)

disaster_ret_df = pd.merge(disaster_ret_df, ret_df[["date", "permno", "ret"]],
                             left_on = ["permno", "month_lead"], 
                             right_on = ["permno", "date"], how = "left")

disaster_ret_df.to_csv("estimated_data/merged_disaster_ret_data/disaster_ret_filter_30.csv")

############################################################
# Doing sorting/trading strategy exercise:
var_to_sort = "D_clamp"
var_to_sort = "rn_prob_20mon"

# 1. Calculate quantiles for each month:
# Calculate outliers that we are not going to invest in:
quant_low_out = disaster_ret_df.groupby("month_lead")[var_to_sort].quantile(0.01).rename("quant_low_out")
quant_high_out = disaster_ret_df.groupby("month_lead")[var_to_sort].quantile(0.99).rename("quant_high_out")

quant_low = disaster_ret_df.groupby("month_lead")[var_to_sort].quantile(0.3).rename("quant_low")
quant_high = disaster_ret_df.groupby("month_lead")[var_to_sort].quantile(0.7).rename("quant_high")

# 2. Merging on original dataframe and assigning buckets:
disaster_ret_quants_df = disaster_ret_df[["month_lead", var_to_sort, "ret"]]
disaster_ret_quants_df = pd.merge(disaster_ret_quants_df, quant_low, on = "month_lead", how = "inner")
disaster_ret_quants_df = pd.merge(disaster_ret_quants_df, quant_high, on = "month_lead", how = "inner")
disaster_ret_quants_df = pd.merge(disaster_ret_quants_df, quant_low_out, on = "month_lead", how = "inner")
disaster_ret_quants_df = pd.merge(disaster_ret_quants_df, quant_high_out, on = "month_lead", how = "inner")

disaster_ret_quants_df["action"] = 0
disaster_ret_quants_df.loc[(disaster_ret_quants_df[var_to_sort] < disaster_ret_quants_df["quant_low"]) & 
                           (disaster_ret_quants_df[var_to_sort] > disaster_ret_quants_df["quant_low_out"]),
           "action"] = -1
disaster_ret_quants_df.loc[(disaster_ret_quants_df[var_to_sort] > disaster_ret_quants_df["quant_high"]) & 
                           (disaster_ret_quants_df[var_to_sort] < disaster_ret_quants_df["quant_high_out"]),
           "action"] = 1

# 3. Doing equal weighted return based on action:
disaster_ret_quants_df["ret_strategy"] = disaster_ret_quants_df["ret"] * disaster_ret_quants_df["action"]
strategy_ret = disaster_ret_quants_df.groupby("month_lead")["ret_strategy"].mean()
gross_ret = strategy_ret + 1
cumprod = gross_ret.cumprod()
cumprod.plot()

############################################################
# Comparing with other Fama-French portfolios:
ff_df = pd.read_csv("estimated_data/final_regression_dfs/ff_factors.csv")
ff_df["date"] = [str(x) + "01" for x in ff_df["date"]]
ff_df["date"] = pd.to_datetime(ff_df["date"], format = "%Y%m%d")
ff_df["date"] = ff_df["date"] + pd.offsets.MonthEnd(0)
for i in range(len(ff_df.columns) - 1):
    ff_df.iloc[:,i+1] = ff_df.iloc[:,i+1]/100
ff_df = ff_df.set_index("date")
    
# 4. Merging with strategy returns data:
strategy_ret = pd.merge(strategy_ret, ff_df, left_index = True, right_index = True)
strategy_ret = strategy_ret.drop("RF", axis = 1)

# 6. Calculating cumulative log return:
gross_ret = strategy_ret + 1
log_ret = np.log(gross_ret)

# 5. Dividing each column by its strandard deviation (to make them comparable):
for i in range(len(log_ret.columns)):
    log_ret.iloc[:,i] = log_ret.iloc[:,i]/(np.std(log_ret.iloc[:,i]))

log_ret.cumsum().plot()

############################################################
# Regressing strategy return on FF portfolios:
# Regressing return of strategy on (1) Market, (2) 3 factors and (3) 5 factors
strategy_ret = strategy_ret.rename({"Mkt-RF": "MKT"}, axis = 1)

results1 = smf.ols(formula = "ret_strategy ~ MKT", data = strategy_ret*12).fit()
results2 = smf.ols(formula = "ret_strategy ~ MKT + SMB + HML", data = strategy_ret*12).fit()
results3 = smf.ols(formula = "ret_strategy ~ MKT  + SMB + HML + CMA + RMW", data = strategy_ret*12).fit()

info_dict={'R-squared' : lambda x: f"{x.rsquared:.2f}",
           'No. observations' : lambda x: f"{int(x.nobs):d}"}

print(summary_col([results1,results2,results3], stars=False, float_format='%0.4f',
                  regressor_order = ["Intercept", "MKT", "SMB", "HML", "CMA", "RMW"],
                  info_dict = info_dict))


