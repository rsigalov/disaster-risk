"""
Company level regressions

This file merges 30 day interpolated D-clamp on secid level with returns data
from CRSP. It then runs a regression of t+1 month return of company i on t month
average D-clamo for company i 
"""

import numpy as np
import pandas as pd
from pandasql import sqldf # for accessing pandas with SQL queries
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats import sandwich_covariance
from statsmodels.iolib.summary2 import summary_col # Fo summarizing regression results

# Loading data:
D_df = pd.read_csv("estimated_data/interpolated_D/int_D_clamp_days_30.csv")
ret_df = pd.read_csv("estimated_data/crsp_data/crsp_monthly_returns.csv")
oclink = pd.read_csv("estimated_data/crsp_data/optionmetrics_crsp_link.csv")

# Calculating average D-clamp for each month and replacing the old dataset:
D_df["date"] = pd.to_datetime(D_df["date"])
D_df["date_adj"] = D_df["date"] + pd.offsets.MonthEnd(0)
D_mean = D_df.groupby(["secid", "date_adj"])["D_clamp"].mean()
D_mean = D_mean.reset_index()

# Getting the best link for each month end of D-clamp:
oclink = oclink[oclink.score < 6]
oclink["sdate"] = [str(int(x)) for x in oclink["sdate"]]
oclink["sdate"] = pd.to_datetime(oclink["sdate"], format = "%Y%m%d")
oclink["edate"] = [str(int(x)) for x in oclink["edate"]]
oclink["edate"] = pd.to_datetime(oclink["edate"], format = "%Y%m%d")

# Merging on 
pysqldf = lambda q: sqldf(q, globals())

q1 = """
select 
    d.secid, d.D_clamp, d.date_adj, 
    s1.permno as permno_1, 
    s2.permno as permno_2,
    s3.permno as permno_3,
    s4.permno as permno_4,
    s5.permno as permno_5
from D_mean as d
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
    secid, date_adj as date, D_clamp,
    COALESCE(permno_1, permno_2, permno_3, permno_4, permno_5) as permno
from tmp
"""

oclink_best_match = pysqldf(q2)

# Merging with returns next month:
oclink_best_match["date"] = pd.to_datetime(oclink_best_match["date"])
oclink_best_match["month_lead"] = oclink_best_match["date"] + pd.offsets.MonthEnd(1)
ret_df["date"] = pd.to_datetime(ret_df["date"])
ret_df["date"] = ret_df["date"] + pd.offsets.MonthEnd(0)
oclink_best_match = pd.merge(oclink_best_match, ret_df[["date", "permno", "ret"]],
                             left_on = ["permno", "month_lead"], 
                             right_on = ["permno", "date"], how = "left")

# Running regressions:
results_all = smf.ols(formula = 'ret ~ D_clamp', data = oclink_best_match).fit()


########################################################################################
# Limiting the universe to (1) companies-months with at least 15 days of observations
# and then to companies that are present in 80% of months
########################################################################################

# Loading data:
D_df = pd.read_csv("estimated_data/interpolated_D/int_D_clamp_days_30.csv")

var = "D_clamp"
D_df["date"] = pd.to_datetime(D_df["date"])
D_df["date_adj"] = D_df["date"] + pd.offsets.MonthEnd(0)

def min_month_obs(x):
    return x[var].count() > 15

D_filter = D_df.groupby(["secid", "date_adj"]).filter(min_month_obs)
D_mon_mean = D_filter.groupby(["secid", "date_adj"])[var].mean().reset_index()

num_months = len(np.unique(D_mon_mean["date_adj"]))
def min_sample_obs(x):
    return x[var].count() > num_months * 0.8

D_mon_mean_filter = D_mon_mean.groupby("secid").filter(min_sample_obs)

# Merging with returns data
pysqldf = lambda q: sqldf(q, globals())
tmp = pysqldf(q1.replace("D_mean", "D_mon_mean_filter"))

# Filtering and providing the best match:
oclink_best_match = pysqldf(q2)

# Merging with returns next month:
oclink_best_match["date"] = pd.to_datetime(oclink_best_match["date"])
oclink_best_match["month_lead"] = oclink_best_match["date"] + pd.offsets.MonthEnd(1)
ret_df["date"] = pd.to_datetime(ret_df["date"])
ret_df["date"] = ret_df["date"] + pd.offsets.MonthEnd(0)
oclink_best_match = pd.merge(oclink_best_match, ret_df[["date", "permno", "ret"]],
                             left_on = ["permno", "month_lead"], 
                             right_on = ["permno", "date"], how = "left")

# Running regressions:
results_filter = smf.ols(formula = 'ret ~ D_clamp', data = oclink_best_match).fit()


# Putting results together:
se_robust_all = np.sqrt(np.diag(sandwich_covariance.cov_hc1(results_all)))[1]
se_robust_filter = np.sqrt(np.diag(sandwich_covariance.cov_hc1(results_filter)))[1]

sample_list = ["all", "all", "filter", "filter"]
se_type_list = ["nonrobust", "robust"] * 2
coef_list = [results_all.params[1], results_all.params[1], 
             results_filter.params[1], results_filter.params[1]]
se_list = [results_all.bse[1], se_robust_all, results_filter.bse[1], se_robust_filter]
N_list = [int(results_all.nobs),int(results_all.nobs), 
          int(results_filter.nobs), int(results_filter.nobs)]

df_secid_level_results = pd.DataFrame({"sample": sample_list,
                                       "se_type": se_type_list,
                                       "coef": coef_list,
                                       "se": se_list,
                                       "N": N_list})

################################################################################
# Now we do predictive regressions using the probability of a large decline
# interpolated to a 30 day level from existing options
################################################################################

# Loading data:
D_df = pd.read_csv("estimated_data/interpolated_D/int_D_clamp_days_30.csv")
ret_df = pd.read_csv("estimated_data/crsp_data/crsp_monthly_returns.csv")
oclink = pd.read_csv("estimated_data/crsp_data/optionmetrics_crsp_link.csv")

# Getting the best link for each month end of D-clamp:
oclink = oclink[oclink.score < 6]
oclink["sdate"] = [str(int(x)) for x in oclink["sdate"]]
oclink["sdate"] = pd.to_datetime(oclink["sdate"], format = "%Y%m%d")
oclink["edate"] = [str(int(x)) for x in oclink["edate"]]
oclink["edate"] = pd.to_datetime(oclink["edate"], format = "%Y%m%d")

# Calculating average D-clamp for each month and replacing the old dataset:
D_df["date"] = pd.to_datetime(D_df["date"])
D_df["date_adj"] = D_df["date"] + pd.offsets.MonthEnd(0)

# Calculating average monthly 

D_mean = D_df.groupby(["secid", "date_adj"])[["rn_prob_20mon", "rn_prob_40mon"]].mean()
D_mean = D_mean.reset_index()

# Merging on 
tmp = pysqldf(q1.replace("d.D_clamp", "d.rn_prob_20mon, d.rn_prob_40mon"))

# Filtering and providing the best match:
oclink_best_match = pysqldf(q2.replace("D_clamp", "rn_prob_20mon, rn_prob_40mon"))

oclink_best_match["date"] = pd.to_datetime(oclink_best_match["date"])
oclink_best_match["month_lead"] = oclink_best_match["date"] + pd.offsets.MonthEnd(1)
ret_df["date"] = pd.to_datetime(ret_df["date"])
ret_df["date"] = ret_df["date"] + pd.offsets.MonthEnd(0)

reg_df = pd.merge(oclink_best_match[["permno", "secid", "month_lead", "rn_prob_20mon", "rn_prob_40mon"]], ret_df[["date", "permno", "ret"]],
                             left_on = ["permno", "month_lead"], 
                             right_on = ["permno", "date"], how = "left")

reg_df = reg_df[(~reg_df["ret"].isnull()) & (~reg_df[var].isnull())]
reg_df = reg_df.drop(columns = ["date"])
reg_df.to_csv("estimated_data/final_regression_dfs/rn_prob_decline_ret.csv")

# Regression in levels:
res = smf.ols(formula = 'ret ~ ' + var, data = reg_df).fit(cov_type = "HC1")
res.summary()

# Creating a dummy of 20% decline:
reg_df["dummy_ret_neg_20"] = np.where(reg_df['ret'] <= -0.2, 1, 0)
res = smf.ols(formula = 'dummy_ret_neg_20 ~ ' + var, data = reg_df).fit(cov_type = "HC1")
res.summary()

# Calculating clustered standard errors:
cov_hetero = sandwich_covariance.cov_hc1(res) # Replicates robust option in STATA
cov_cluster_firm = sandwich_covariance.cov_cluster(res, reg_df['secid']) # Matches with one in STATA
cov_cluster_month = sandwich_covariance.cov_cluster(res, reg_df['month_lead']) # Matches with one in STATA

#### Double clustering by firm and year ####
# To do it we need to calculate the covariance matrix with clustering
# by firm and by time and add them together, then subtract covariance matrix 
# with Heteroskedascity correction
cov_double_cluster = cov_cluster_firm + cov_cluster_month - cov_hetero

np.sqrt(np.diag(cov_double_cluster))

# Excluding 2007, 2008 and 2009:
reg_df_excl_crisis = reg_df[(reg_df["month_lead"] < "2007-01-01") | (reg_df["month_lead"] > "2009-12-31")]
reg_df_excl_crisis["dummy_ret_neg_20"] = np.where(reg_df_excl_crisis['ret'] <= -0.2, 1, 0)
res = smf.ols(formula = 'dummy_ret_neg_20 ~ ' + var, data = reg_df_excl_crisis).fit(cov_type = "HC1")
res.summary()

######################################################################################
# Limiting sample to companies with 80% of the months that are sufficiently populated
######################################################################################

def min_month_obs(x):
    return x[var].count() > 15

D_filter = D_df.groupby(["secid", "date_adj"]).filter(min_month_obs)
D_mon_mean = D_filter.groupby(["secid", "date_adj"])[["rn_prob_20mon", "rn_prob_40mon"]].mean().reset_index()

num_months = len(np.unique(D_mon_mean["date_adj"]))
def min_sample_obs(x):
    return x["rn_prob_40mon"].count() > num_months * 0.8

D_mon_mean_filter = D_mon_mean.groupby("secid").filter(min_sample_obs)

# Merging with returns data
tmp = pysqldf(q1.replace("D_mean", "D_mon_mean_filter").replace("d.D_clamp", "d.rn_prob_20mon, d.rn_prob_40mon"))
oclink_best_match = pysqldf(q2.replace("D_clamp", "rn_prob_20mon, rn_prob_40mon"))

# Merging returns with 1 month lead for predictive regression:
oclink_best_match["date"] = pd.to_datetime(oclink_best_match["date"])
oclink_best_match["month_lead"] = oclink_best_match["date"] + pd.offsets.MonthEnd(1)
ret_df["date"] = pd.to_datetime(ret_df["date"])
ret_df["date"] = ret_df["date"] + pd.offsets.MonthEnd(0)

reg_filter_df = pd.merge(oclink_best_match[["permno", "secid", "month_lead", "rn_prob_20mon", "rn_prob_40mon"]], ret_df[["date", "permno", "ret"]],
                             left_on = ["permno", "month_lead"], 
                             right_on = ["permno", "date"], how = "left")

reg_filter_df = reg_filter_df[(~reg_filter_df["ret"].isnull()) & (~reg_filter_df[var].isnull())]
reg_filter_df = reg_filter_df.drop(columns = ["date"])
reg_filter_df.to_csv("estimated_data/final_regression_dfs/rn_prob_decline_ret_filter.csv")

reg_filter_df["dummy_ret_neg_20"] = np.where(reg_filter_df['ret'] <= -0.2, 1, 0)
res = smf.ols(formula = 'dummy_ret_neg_20 ~ ' + var, data = reg_filter_df).fit(cov_type = "HC1")
res.summary()

# Calculating clustered standard errors:
cov_hetero = sandwich_covariance.cov_hc1(res) # Replicates robust option in STATA
cov_cluster_firm = sandwich_covariance.cov_cluster(res, reg_filter_df['secid']) # Matches with one in STATA
cov_cluster_month = sandwich_covariance.cov_cluster(res, reg_filter_df['month_lead']) # Matches with one in STATA

# Double clustering by firm and year #
cov_double_cluster = cov_cluster_firm + cov_cluster_month - cov_hetero


# Merging filtered dataset on the full dataset to create a dummy for filtered
# month-secid:
reg_filter_df["dummy_filter"] = 1
reg_df = pd.merge(reg_df, reg_filter_df[["secid", "month_lead", "dummy_filter"]],
                  on = ["secid", "month_lead"], how = "left")

reg_df.loc[reg_df["dummy_filter"].isnull(), "dummy_filter"] = 0
reg_df.drop(columns = "dummy_ret_neg_20").to_csv("estimated_data/final_regression_dfs/rn_prob_decline_ret.csv", index = False)






