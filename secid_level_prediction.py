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

# Loading data:
D_df = pd.read_csv("estimated_data/interpolated_D/int_D_clamp_days_30.csv")
ret_df = pd.read_csv("estimated_data/crsp_data/crsp_monthly_returns.csv")
oclink = pd.read_csv("estimated_data/crsp_data/optionmetrics_crsp_link.csv")

# Calculating average D-clamp for each month and replacing the old dataset:
D_df["date"] = pd.to_datetime(D_df["date"])
D_df["date_adj"] = D_df["date"] + pd.offsets.MonthEnd(0)
D_df = D_df.groupby(["secid", "date_adj"])["D_clamp"].mean()
D_df = D_df.reset_index()

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
from D_df as d
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

q1 = """
select 
    d.secid, d.D_clamp, d.date_adj, 
    s1.permno as permno_1, 
    s2.permno as permno_2,
    s3.permno as permno_3,
    s4.permno as permno_4,
    s5.permno as permno_5
from D_mon_mean_filter as d
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















